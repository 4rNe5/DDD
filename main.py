import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QFileDialog, QPushButton, QSlider, QLabel,
                             QGroupBox, QMessageBox, QColorDialog, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap, QImage, QPainter
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *


class Outline3DWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.rotation_x = 0
        self.rotation_y = 0
        self.scale = 1.0
        self.translate_z = -300
        self.last_x = 0
        self.last_y = 0
        self.mouse_pressed = False
        self.outlines = []  # 외곽선 점들
        self.extrusion_depth = 10.0  # 돌출 깊이
        self.line_color = QColor(25, 25, 25)  # 기본 색상
        self.wireframe_mode = True
        self.has_image = False
        self.threshold_value = 127
        self.show_inner_contours = True

    def load_image_and_extract_outline(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("이미지를 읽을 수 없습니다")

            self.original_image = img.copy()
            self.extract_outlines()
            return True
        except Exception as e:
            print(f"이미지 로딩 오류: {e}")
            return False

    def extract_outlines(self):
        img = self.original_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 노이즈 제거를 위한 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 경계선 감지를 위한 이진화
        ret, thresh = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY_INV)

        # 컨투어 찾기 (계층 구조 포함)
        contour_mode = cv2.RETR_TREE if self.show_inner_contours else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(thresh, contour_mode, cv2.CHAIN_APPROX_SIMPLE)

        # 작은 컨투어 필터링 (노이즈 제거)
        min_contour_area = 100  # 최소 영역 크기
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        self.outlines = []
        h, w = gray.shape
        for cnt in filtered_contours:
            # 컨투어 단순화 (더 부드러운 라인)
            epsilon = 0.0025 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            points = []
            for pt in approx:
                x, y = pt[0]
                # 중심 정렬 및 y축 뒤집기(화면 좌표계와 OpenGL 좌표계 맞춤)
                points.append([x - w / 2, -(y - h / 2), 0])
            self.outlines.append(points)

        self.has_image = True
        self.update()

    def set_threshold(self, value):
        self.threshold_value = value
        if hasattr(self, 'original_image'):
            self.extract_outlines()

    def toggle_inner_contours(self, state):
        self.show_inner_contours = state == Qt.Checked
        if hasattr(self, 'original_image'):
            self.extract_outlines()

    def initializeGL(self):
        glClearColor(1, 1, 1, 1)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glLineWidth(2.0)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height if height else 1, 1, 1000)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, 0, self.translate_z)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glScalef(self.scale, self.scale, self.scale)

        if not self.has_image:
            return

        # 외곽선 색상 설정
        color = self.line_color.getRgbF()
        glColor3f(color[0], color[1], color[2])

        for outline in self.outlines:
            if len(outline) < 3:
                continue

            if self.wireframe_mode:
                self.draw_wireframe(outline)
            else:
                self.draw_solid(outline)

    def draw_wireframe(self, outline):
        # 앞면 그리기
        glBegin(GL_LINE_LOOP)
        for pt in outline:
            glVertex3f(pt[0], pt[1], 0)
        glEnd()

        # 뒷면 그리기
        glBegin(GL_LINE_LOOP)
        for pt in outline:
            glVertex3f(pt[0], pt[1], -self.extrusion_depth)
        glEnd()

        # 면 사이 연결선 그리기
        glBegin(GL_LINES)
        for pt in outline:
            glVertex3f(pt[0], pt[1], 0)
            glVertex3f(pt[0], pt[1], -self.extrusion_depth)
        glEnd()

    def draw_solid(self, outline):
        # 앞면
        glBegin(GL_POLYGON)
        for pt in outline:
            glVertex3f(pt[0], pt[1], 0)
        glEnd()

        # 뒷면
        glBegin(GL_POLYGON)
        for pt in reversed(outline):
            glVertex3f(pt[0], pt[1], -self.extrusion_depth)
        glEnd()

        # 옆면
        glBegin(GL_QUAD_STRIP)
        for i in range(len(outline)):
            pt = outline[i]
            glVertex3f(pt[0], pt[1], 0)
            glVertex3f(pt[0], pt[1], -self.extrusion_depth)

        # 루프 닫기
        pt = outline[0]
        glVertex3f(pt[0], pt[1], 0)
        glVertex3f(pt[0], pt[1], -self.extrusion_depth)
        glEnd()

    def set_extrusion_depth(self, depth):
        self.extrusion_depth = depth
        self.update()

    def set_color(self, color):
        self.line_color = color
        self.update()

    def toggle_render_mode(self):
        self.wireframe_mode = not self.wireframe_mode
        self.update()

    def mousePressEvent(self, event):
        self.mouse_pressed = True
        self.last_x = event.x()
        self.last_y = event.y()

    def mouseMoveEvent(self, event):
        if self.mouse_pressed:
            dx = event.x() - self.last_x
            dy = event.y() - self.last_y
            self.rotation_x += dy
            self.rotation_y += dx
            self.last_x = event.x()
            self.last_y = event.y()
            self.update()

    def mouseReleaseEvent(self, event):
        self.mouse_pressed = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        # 줌 속도를 현재 스케일에 비례하게 조정
        zoom_factor = 1.1

        # 줌인
        if delta > 0:
            self.scale *= zoom_factor
            # 최대 줌 제한
            if self.scale > 20.0:
                self.scale = 20.0
        # 줌아웃
        else:
            self.scale /= zoom_factor
            # 최소 줌 제한
            if self.scale < 0.1:
                self.scale = 0.1

        # 줌 변경 시 깊이 조정 (더 직관적인 줌 효과)
        self.translate_z = -300 * (1.0 / self.scale)

        self.update()


class ImagePreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.image = None

    def set_image(self, image_path):
        self.image = QPixmap(image_path)
        self.update()

    def paintEvent(self, event):
        if self.image:
            painter = QPainter(self)
            # 이미지 크기를 위젯에 맞게 조절
            scaled_img = self.image.scaled(self.size(), Qt.KeepAspectRatio)
            # 중앙에 정렬
            x = (self.width() - scaled_img.width()) // 2
            y = (self.height() - scaled_img.height()) // 2
            painter.drawPixmap(x, y, scaled_img)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('일러스트 외곽선 3D 시각화')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # 상단 컨트롤
        top_controls = QHBoxLayout()

        # 이미지 열기 버튼
        open_btn = QPushButton('이미지 열기')
        open_btn.clicked.connect(self.open_image)
        top_controls.addWidget(open_btn)

        # 색상 선택 버튼
        color_btn = QPushButton('색상 변경')
        color_btn.clicked.connect(self.change_color)
        top_controls.addWidget(color_btn)

        # 와이어프레임/솔리드 전환 버튼
        self.render_btn = QPushButton('와이어프레임/솔리드 전환')
        self.render_btn.clicked.connect(self.toggle_render_mode)
        top_controls.addWidget(self.render_btn)

        # 초기화 버튼
        reset_btn = QPushButton('초기화')
        reset_btn.clicked.connect(self.reset_view)
        top_controls.addWidget(reset_btn)

        # 내보내기 버튼 추가
        export_btn = QPushButton('3D 모델 내보내기')
        export_btn.clicked.connect(self.export_model)
        top_controls.addWidget(export_btn)

        main_layout.addLayout(top_controls)

        # 3D 위젯과 컨트롤
        content_layout = QHBoxLayout()

        # 3D 위젯
        self.outline_widget = Outline3DWidget()
        content_layout.addWidget(self.outline_widget, 3)  # 비율 3

        # 사이드바 컨트롤과 미리보기
        sidebar_layout = QVBoxLayout()

        # 컨트롤 그룹
        controls_group = QGroupBox("컨트롤")
        controls_layout = QVBoxLayout()

        # 돌출 깊이 컨트롤
        depth_label = QLabel("돌출 깊이:")
        controls_layout.addWidget(depth_label)

        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setMinimum(1)
        self.depth_slider.setMaximum(100)
        self.depth_slider.setValue(10)
        self.depth_slider.setTickPosition(QSlider.TicksBelow)
        self.depth_slider.valueChanged.connect(self.change_extrusion_depth)
        controls_layout.addWidget(self.depth_slider)

        self.depth_value = QLabel("10")
        controls_layout.addWidget(self.depth_value)

        # 경계값 컨트롤 (이미지 외곽선 감지 임계값)
        threshold_label = QLabel("외곽선 감지 임계값:")
        controls_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(240)
        self.threshold_slider.setValue(127)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.valueChanged.connect(self.change_threshold)
        controls_layout.addWidget(self.threshold_slider)

        self.threshold_value = QLabel("127")
        controls_layout.addWidget(self.threshold_value)

        # 내부 윤곽선 표시 옵션
        self.inner_contours_check = QCheckBox("내부 윤곽선 표시")
        self.inner_contours_check.setChecked(True)
        self.inner_contours_check.stateChanged.connect(self.toggle_inner_contours)
        controls_layout.addWidget(self.inner_contours_check)

        controls_layout.addStretch(1)
        controls_group.setLayout(controls_layout)
        sidebar_layout.addWidget(controls_group)

        # 이미지 미리보기
        preview_group = QGroupBox("원본 이미지")
        preview_layout = QVBoxLayout()
        self.preview_widget = QLabel()
        self.preview_widget.setAlignment(Qt.AlignCenter)
        self.preview_widget.setMinimumHeight(200)
        self.preview_widget.setStyleSheet("background-color: white;")
        preview_layout.addWidget(self.preview_widget)
        preview_group.setLayout(preview_layout)
        sidebar_layout.addWidget(preview_group)

        # 사이드바를 전체 레이아웃에 추가
        content_layout.addLayout(sidebar_layout, 1)  # 비율 1

        main_layout.addLayout(content_layout)

        # 상태 표시줄
        self.statusBar().showMessage('이미지를 열어 시작하세요')

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '이미지 선택',
            '',
            'Images (*.png *.jpg *.jpeg *.bmp *.tiff)'
        )

        if file_path:
            # 로딩 중 메시지 표시
            self.statusBar().showMessage("이미지 처리 중...")
            QApplication.processEvents()

            success = self.outline_widget.load_image_and_extract_outline(file_path)

            if success:
                # 파일 이름 표시
                import os
                filename = os.path.basename(file_path)
                self.statusBar().showMessage(f"로드됨: {filename}")

                # 미리보기 이미지 표시
                pixmap = QPixmap(file_path)
                self.preview_widget.setPixmap(pixmap.scaled(
                    self.preview_widget.width(),
                    self.preview_widget.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))

                # 뷰 초기화
                self.reset_view()
            else:
                self.statusBar().showMessage("이미지 로드 실패")
                QMessageBox.critical(self, "오류", "이미지를 로드하거나 처리하는 데 실패했습니다.")

    def change_color(self):
        color = QColorDialog.getColor(self.outline_widget.line_color, self, "색상 선택")
        if color.isValid():
            self.outline_widget.set_color(color)
            self.statusBar().showMessage(f"색상 변경됨: RGB({color.red()}, {color.green()}, {color.blue()})")

    def change_extrusion_depth(self, value):
        self.depth_value.setText(str(value))
        self.outline_widget.set_extrusion_depth(float(value))
        self.statusBar().showMessage(f"돌출 깊이: {value}")

    def change_threshold(self, value):
        self.threshold_value.setText(str(value))
        self.outline_widget.set_threshold(value)
        self.statusBar().showMessage(f"임계값: {value}")

    def toggle_inner_contours(self, state):
        self.outline_widget.toggle_inner_contours(state)
        status = "표시" if state == Qt.Checked else "숨김"
        self.statusBar().showMessage(f"내부 윤곽선: {status}")

    def toggle_render_mode(self):
        self.outline_widget.toggle_render_mode()
        mode = "솔리드" if not self.outline_widget.wireframe_mode else "와이어프레임"
        self.statusBar().showMessage(f"렌더링 모드: {mode}")

    def reset_view(self):
        # 3D 뷰 초기화
        self.outline_widget.rotation_x = 0
        self.outline_widget.rotation_y = 0
        self.outline_widget.scale = 1.0
        self.outline_widget.translate_z = -300
        self.outline_widget.update()
        self.statusBar().showMessage("뷰 초기화됨")

    def export_model(self):
        if not self.outline_widget.has_image:
            QMessageBox.warning(self, "경고", "내보낼 모델이 없습니다. 먼저 이미지를 로드하세요.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            '3D 모델 저장',
            '',
            'STL Files (*.stl);;OBJ Files (*.obj)'
        )

        if file_path:
            try:
                # 파일 확장자에 따라 내보내기 함수 호출
                if file_path.endswith('.stl'):
                    self.export_to_stl(file_path)
                elif file_path.endswith('.obj'):
                    self.export_to_obj(file_path)
                else:
                    # 확장자가 없는 경우, 선택한 필터에 따라 확장자 추가
                    if selected_filter == 'STL Files (*.stl)':
                        file_path += '.stl'
                        self.export_to_stl(file_path)
                    else:
                        file_path += '.obj'
                        self.export_to_obj(file_path)

                self.statusBar().showMessage(f"모델이 {file_path}에 저장되었습니다")
            except Exception as e:
                self.statusBar().showMessage("내보내기 실패")
                QMessageBox.critical(self, "내보내기 오류", f"3D 모델 내보내기 실패: {str(e)}")

    def export_to_stl(self, filepath):
        """STL 파일로 내보내기 (ASCII 포맷)"""
        with open(filepath, 'w') as f:
            f.write("solid extruded_image\n")

            for outline in self.outline_widget.outlines:
                if len(outline) < 3:
                    continue

                # 삼각형으로 면 분할
                # 간단한 fan triangulation (첫번째 점에서 모든 삼각형 형성)
                for i in range(1, len(outline) - 1):
                    # 앞면 삼각형
                    p1 = outline[0]
                    p2 = outline[i]
                    p3 = outline[i + 1]

                    # 앞면 법선 벡터는 +z 방향
                    f.write(f"facet normal 0.0 0.0 1.0\n")
                    f.write("  outer loop\n")
                    f.write(f"    vertex {p1[0]:.6f} {p1[1]:.6f} {0.0:.6f}\n")
                    f.write(f"    vertex {p2[0]:.6f} {p2[1]:.6f} {0.0:.6f}\n")
                    f.write(f"    vertex {p3[0]:.6f} {p3[1]:.6f} {0.0:.6f}\n")
                    f.write("  endloop\n")
                    f.write("endfacet\n")

                    # 뒷면 삼각형 (법선 벡터 반대)
                    depth = -self.outline_widget.extrusion_depth
                    f.write(f"facet normal 0.0 0.0 -1.0\n")
                    f.write("  outer loop\n")
                    f.write(f"    vertex {p1[0]:.6f} {p1[1]:.6f} {depth:.6f}\n")
                    f.write(f"    vertex {p3[0]:.6f} {p3[1]:.6f} {depth:.6f}\n")
                    f.write(f"    vertex {p2[0]:.6f} {p2[1]:.6f} {depth:.6f}\n")
                    f.write("  endloop\n")
                    f.write("endfacet\n")

                # 옆면 (각 가장자리를 삼각형 2개로 만듦)
                depth = -self.outline_widget.extrusion_depth
                for i in range(len(outline)):
                    p1 = outline[i]
                    p2 = outline[(i + 1) % len(outline)]  # 다음 점 (마지막 점은 첫 점으로 연결)

                    # 대략적인 법선 계산 (정확한 법선 계산은 복잡할 수 있음)
                    nx = p2[1] - p1[1]  # 간단한 외적
                    ny = p1[0] - p2[0]
                    mag = (nx * nx + ny * ny) ** 0.5
                    if mag > 0:
                        nx, ny = nx / mag, ny / mag

                    # 첫 번째 삼각형
                    f.write(f"facet normal {nx:.6f} {ny:.6f} 0.0\n")
                    f.write("  outer loop\n")
                    f.write(f"    vertex {p1[0]:.6f} {p1[1]:.6f} {0.0:.6f}\n")
                    f.write(f"    vertex {p2[0]:.6f} {p2[1]:.6f} {0.0:.6f}\n")
                    f.write(f"    vertex {p1[0]:.6f} {p1[1]:.6f} {depth:.6f}\n")
                    f.write("  endloop\n")
                    f.write("endfacet\n")

                    # 두 번째 삼각형
                    f.write(f"facet normal {nx:.6f} {ny:.6f} 0.0\n")
                    f.write("  outer loop\n")
                    f.write(f"    vertex {p2[0]:.6f} {p2[1]:.6f} {0.0:.6f}\n")
                    f.write(f"    vertex {p2[0]:.6f} {p2[1]:.6f} {depth:.6f}\n")
                    f.write(f"    vertex {p1[0]:.6f} {p1[1]:.6f} {depth:.6f}\n")
                    f.write("  endloop\n")
                    f.write("endfacet\n")

            f.write("endsolid extruded_image\n")

    def export_to_obj(self, filepath):
        """OBJ 파일로 내보내기"""
        with open(filepath, 'w') as f:
            f.write("# 이미지 외곽선에서 생성된 3D 모델\n")

            vertex_count = 1  # OBJ는 1부터 인덱스 시작

            for outline_idx, outline in enumerate(self.outline_widget.outlines):
                if len(outline) < 3:
                    continue

                # 그룹 이름 작성
                f.write(f"g outline_{outline_idx}\n")

                # 앞면 정점
                front_start_idx = vertex_count
                for pt in outline:
                    f.write(f"v {pt[0]:.6f} {pt[1]:.6f} {0.0:.6f}\n")
                    vertex_count += 1

                # 뒷면 정점
                back_start_idx = vertex_count
                depth = -self.outline_widget.extrusion_depth
                for pt in outline:
                    f.write(f"v {pt[0]:.6f} {pt[1]:.6f} {depth:.6f}\n")
                    vertex_count += 1

                # 앞면 (삼각형 팬 형태로 분할)
                f.write("f")
                for i in range(front_start_idx, back_start_idx):
                    f.write(f" {i}")
                f.write("\n")

                # 뒷면 (역순)
                f.write("f")
                for i in range(back_start_idx + len(outline) - 1, back_start_idx - 1, -1):
                    f.write(f" {i}")
                f.write("\n")

                # 옆면 (사각형으로)
                for i in range(len(outline)):
                    idx1 = front_start_idx + i
                    idx2 = front_start_idx + (i + 1) % len(outline)
                    idx3 = back_start_idx + (i + 1) % len(outline)
                    idx4 = back_start_idx + i
                    f.write(f"f {idx1} {idx2} {idx3} {idx4}\n")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

