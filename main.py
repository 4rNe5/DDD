import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Rectangle, Polygon
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading


class ImageAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("이미지 그래프 분석기")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # 변수들
        self.image_path = None
        self.original_image = None
        self.analyzer = None

        # GUI 설정
        self.setup_gui()

    def setup_gui(self):
        """GUI Component Set-up"""
        # 메인 프레임
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 상단 제어 패널
        control_frame = tk.Frame(main_frame, bg='#e0e0e0', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 제목
        title_label = tk.Label(control_frame, text="🎨 이미지 그래프 분석기",
                               font=('Arial', 16, 'bold'), bg='#e0e0e0')
        title_label.pack(pady=10)

        # 버튼 프레임
        button_frame = tk.Frame(control_frame, bg='#e0e0e0')
        button_frame.pack(pady=10)

        # 이미지 선택 버튼
        self.select_btn = tk.Button(button_frame, text="📁 이미지 선택",
                                    command=self.select_image, font=('Arial', 10, 'bold'),
                                    bg='#4CAF50', fg='white', width=15, height=2)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        # 테스트 이미지 생성 버튼
        self.test_btn = tk.Button(button_frame, text="🧪 테스트 이미지 생성",
                                  command=self.create_test_image, font=('Arial', 10, 'bold'),
                                  bg='#2196F3', fg='white', width=15, height=2)
        self.test_btn.pack(side=tk.LEFT, padx=5)

        # 선택된 파일 표시
        self.file_label = tk.Label(control_frame, text="선택된 파일: 없음",
                                   font=('Arial', 10), bg='#e0e0e0')
        self.file_label.pack(pady=5)

        # 분석 방법 선택 프레임
        analysis_frame = tk.LabelFrame(control_frame, text="분석 방법 선택",
                                       font=('Arial', 12, 'bold'), bg='#e0e0e0')
        analysis_frame.pack(fill=tk.X, padx=10, pady=10)

        # 분석 버튼들
        analysis_buttons = [
            ("🔍 엣지 검출", self.edge_analysis, '#FF5722'),
            ("📐 윤곽선 분석", self.contour_analysis, '#9C27B0'),
            ("🎨 색상 클러스터링", self.color_analysis, '#E91E63'),
            ("📊 픽셀 강도 분석", self.intensity_analysis, '#FF9800'),
            ("🔸 기하학적 형태", self.geometric_analysis, '#607D8B'),
            ("📈 전체 분석", self.analyze_all, '#F44336')
        ]

        btn_frame = tk.Frame(analysis_frame, bg='#e0e0e0')
        btn_frame.pack(pady=10)

        for i, (text, command, color) in enumerate(analysis_buttons):
            btn = tk.Button(btn_frame, text=text, command=command,
                            font=('Arial', 9, 'bold'), bg=color, fg='white',
                            width=18, height=2, state=tk.DISABLED)
            btn.grid(row=i // 3, column=i % 3, padx=5, pady=5)
            setattr(self, f'analysis_btn_{i}', btn)

        # 진행 상황 표시
        self.progress_frame = tk.Frame(control_frame, bg='#e0e0e0')
        self.progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress_label = tk.Label(self.progress_frame, text="",
                                       font=('Arial', 10), bg='#e0e0e0')
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        # 하단 결과 표시 영역
        self.result_frame = tk.Frame(main_frame, bg='#f0f0f0')
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        # 이미지 미리보기 프레임
        self.preview_frame = tk.LabelFrame(self.result_frame, text="이미지 미리보기",
                                           font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.preview_label = tk.Label(self.preview_frame, text="이미지를 선택해주세요",
                                      font=('Arial', 12), bg='white', relief=tk.SUNKEN)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 분석 결과 프레임
        self.analysis_frame = tk.LabelFrame(self.result_frame, text="분석 결과",
                                            font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.analysis_label = tk.Label(self.analysis_frame, text="분석 결과가 여기에 표시됩니다",
                                       font=('Arial', 12), bg='white', relief=tk.SUNKEN)
        self.analysis_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def select_image(self):
        """이미지 파일 선택"""
        file_types = [
            ('이미지 파일', '*.jpg *.jpeg *.png *.bmp *.tiff *.gif'),
            ('JPEG 파일', '*.jpg *.jpeg'),
            ('PNG 파일', '*.png'),
            ('모든 파일', '*.*')
        ]

        self.image_path = filedialog.askopenfilename(
            title="분석할 이미지를 선택하세요",
            filetypes=file_types
        )

        if self.image_path:
            self.load_image()

    def create_test_image(self):
        """테스트 이미지 생성"""
        try:
            # 테스트 이미지 생성
            test_img = np.zeros((300, 300, 3), dtype=np.uint8)
            test_img.fill(255)  # 흰색 배경

            # 다양한 도형 그리기
            cv2.rectangle(test_img, (50, 50), (120, 120), (255, 100, 100), -1)  # 빨간 사각형
            cv2.circle(test_img, (200, 80), 30, (100, 255, 100), -1)  # 초록 원
            cv2.ellipse(test_img, (150, 180), (40, 25), 0, 0, 360, (100, 100, 255), -1)  # 파란 타원

            # 삼각형
            pts = np.array([[80, 200], [120, 200], [100, 160]], np.int32)
            cv2.fillPoly(test_img, [pts], (255, 255, 100))

            # 선 그리기
            cv2.line(test_img, (50, 250), (250, 250), (200, 100, 200), 3)

            # 작은 노이즈 추가
            noise = np.random.randint(0, 30, test_img.shape, dtype=np.uint8)
            test_img = cv2.add(test_img, noise)

            # 파일 저장
            test_path = "test_image_gui.jpg"
            cv2.imwrite(test_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))

            self.image_path = test_path
            self.load_image()

            messagebox.showinfo("완료", "테스트 이미지가 생성되었습니다!")

        except Exception as e:
            messagebox.showerror("오류", f"테스트 이미지 생성 실패: {str(e)}")

    def load_image(self):
        """이미지 로드 및 미리보기 표시"""
        try:
            # 이미지 로드
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                raise ValueError("이미지를 로드할 수 없습니다")

            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

            # 분석기 초기화
            self.analyzer = ImageToGraphAnalyzer(self.image_path)

            # 파일명 표시
            filename = os.path.basename(self.image_path)
            self.file_label.config(text=f"선택된 파일: {filename}")

            # 미리보기 이미지 표시
            self.show_preview()

            # 분석 버튼들 활성화
            for i in range(6):
                btn = getattr(self, f'analysis_btn_{i}')
                btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패: {str(e)}")

    def show_preview(self):
        """이미지 미리보기 표시"""
        if self.original_image is not None:
            # 이미지 크기 조정
            height, width = self.original_image.shape[:2]
            max_size = 300

            if height > max_size or width > max_size:
                scale = min(max_size / height, max_size / width)
                new_height, new_width = int(height * scale), int(width * scale)
                resized = cv2.resize(self.original_image, (new_width, new_height))
            else:
                resized = self.original_image.copy()

            # PIL 이미지로 변환
            pil_img = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(pil_img)

            # 미리보기 라벨에 표시
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo  # 참조 유지

    def show_progress(self, text):
        """진행 상황 표시"""
        self.progress_label.config(text=text)
        self.progress_bar.start()
        self.root.update()

    def hide_progress(self):
        """진행 상황 숨기기"""
        self.progress_label.config(text="")
        self.progress_bar.stop()
        self.root.update()

    def show_result_in_window(self, fig, title):
        """결과를 새 창에 표시"""
        result_window = tk.Toplevel(self.root)
        result_window.title(title)
        result_window.geometry("1000x700")

        # matplotlib 캔버스 생성
        canvas = FigureCanvasTkAgg(fig, result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 저장 버튼
        save_btn = tk.Button(result_window, text="📁 결과 저장",
                             command=lambda: self.save_result(fig, title),
                             font=('Arial', 10, 'bold'), bg='#4CAF50', fg='white')
        save_btn.pack(pady=5)

    def save_result(self, fig, title):
        """분석 결과 저장"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                title=f"{title} 저장"
            )
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("완료", f"결과가 저장되었습니다: {filename}")
        except Exception as e:
            messagebox.showerror("오류", f"저장 실패: {str(e)}")

    def run_analysis(self, analysis_func, title):
        """분석 실행 (별도 스레드)"""

        def analysis_thread():
            try:
                self.show_progress(f"{title} 분석 중...")
                fig = analysis_func()
                self.hide_progress()

                # 메인 스레드에서 결과 표시
                self.root.after(0, lambda: self.show_result_in_window(fig, title))

            except Exception as e:
                self.hide_progress()
                self.root.after(0, lambda: messagebox.showerror("오류", f"{title} 분석 실패: {str(e)}"))

        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()

    def edge_analysis(self):
        """엣지 검출 분석"""
        if self.analyzer:
            self.run_analysis(self.analyzer.edge_detection_graph, "엣지 검출")

    def contour_analysis(self):
        """윤곽선 분석"""
        if self.analyzer:
            self.run_analysis(self.analyzer.contour_graph, "윤곽선 분석")

    def color_analysis(self):
        """색상 클러스터링 분석"""
        if self.analyzer:
            self.run_analysis(self.analyzer.color_clustering_graph, "색상 클러스터링")

    def intensity_analysis(self):
        """픽셀 강도 분석"""
        if self.analyzer:
            self.run_analysis(self.analyzer.pixel_intensity_graph, "픽셀 강도 분석")

    def geometric_analysis(self):
        """기하학적 형태 분석"""
        if self.analyzer:
            self.run_analysis(self.analyzer.geometric_shape_graph, "기하학적 형태 분석")

    def analyze_all(self):
        """전체 분석"""
        if self.analyzer:
            def all_analysis():
                try:
                    self.show_progress("전체 분석 중...")

                    # 모든 분석 실행
                    analyses = [
                        (self.analyzer.edge_detection_graph, "엣지 검출"),
                        (self.analyzer.contour_graph, "윤곽선 분석"),
                        (self.analyzer.color_clustering_graph, "색상 클러스터링"),
                        (self.analyzer.pixel_intensity_graph, "픽셀 강도 분석"),
                        (self.analyzer.geometric_shape_graph, "기하학적 형태 분석")
                    ]

                    for i, (func, title) in enumerate(analyses):
                        self.root.after(0, lambda t=title: self.progress_label.config(text=f"{t} 분석 중..."))
                        fig = func()
                        self.root.after(0, lambda f=fig, t=title: self.show_result_in_window(f, t))

                    self.hide_progress()
                    self.root.after(0, lambda: messagebox.showinfo("완료", "모든 분석이 완료되었습니다!"))

                except Exception as e:
                    self.hide_progress()
                    self.root.after(0, lambda: messagebox.showerror("오류", f"전체 분석 실패: {str(e)}"))

            thread = threading.Thread(target=all_analysis)
            thread.daemon = True
            thread.start()


class ImageToGraphAnalyzer:
    """기존 분석 클래스 (GUI 버전에서 사용)"""

    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.load_image()

    def load_image(self):
        """이미지 로드"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {self.image_path}")
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

    def edge_detection_graph(self):
        """엣지 검출을 통한 선 그래프 생성"""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        edge_points = np.where(edges == 255)
        y_coords = edge_points[0]
        x_coords = edge_points[1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.imshow(self.original_image)
        ax1.set_title('원본 이미지', fontsize=14)
        ax1.axis('off')

        ax2.scatter(x_coords, y_coords, s=0.5, c='black', alpha=0.8)
        ax2.set_xlim(0, self.original_image.shape[1])
        ax2.set_ylim(self.original_image.shape[0], 0)
        ax2.set_title('엣지 검출 그래프', fontsize=14)
        ax2.set_aspect('equal')

        plt.tight_layout()
        return fig

    def contour_graph(self):
        """윤곽선을 이용한 그래프 생성"""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.imshow(self.original_image)
        ax1.set_title('원본 이미지', fontsize=14)
        ax1.axis('off')

        ax2.set_xlim(0, self.original_image.shape[1])
        ax2.set_ylim(self.original_image.shape[0], 0)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(contours)))
        for contour, color in zip(contours, colors):
            if len(contour) > 5:
                x_coords = contour[:, 0, 0]
                y_coords = contour[:, 0, 1]
                ax2.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)

        ax2.set_title('윤곽선 그래프', fontsize=14)
        ax2.set_aspect('equal')

        plt.tight_layout()
        return fig

    def color_clustering_graph(self, n_colors=8):
        """색상 클러스터링을 통한 그래프 생성"""
        height, width = self.original_image.shape[:2]
        if height > 300 or width > 300:
            scale = min(300 / height, 300 / width)
            new_height, new_width = int(height * scale), int(width * scale)
            resized = cv2.resize(self.original_image, (new_width, new_height))
        else:
            resized = self.original_image.copy()

        data = resized.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        centers = kmeans.cluster_centers_

        clustered_data = centers[labels]
        clustered_image = clustered_data.reshape(resized.shape).astype(np.uint8)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('원본 이미지', fontsize=14)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(clustered_image)
        axes[0, 1].set_title(f'{n_colors}색으로 클러스터링', fontsize=14)
        axes[0, 1].axis('off')

        unique_labels = np.unique(labels)
        colors = centers[unique_labels] / 255.0
        counts = [np.sum(labels == i) for i in unique_labels]

        axes[1, 0].pie(counts, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('색상 분포', fontsize=14)

        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        sample_indices = np.random.choice(len(data), min(1000, len(data)), replace=False)
        sample_data = data[sample_indices]
        sample_colors = sample_data / 255.0

        ax_3d.scatter(sample_data[:, 0], sample_data[:, 1], sample_data[:, 2],
                      c=sample_colors, s=2, alpha=0.6)
        ax_3d.set_xlabel('Red')
        ax_3d.set_ylabel('Green')
        ax_3d.set_zlabel('Blue')
        ax_3d.set_title('RGB 색상 공간', fontsize=14)

        plt.tight_layout()
        return fig

    def pixel_intensity_graph(self):
        """픽셀 강도 분석 그래프"""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('원본 이미지', fontsize=14)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('그레이스케일', fontsize=14)
        axes[0, 1].axis('off')

        axes[1, 0].plot(hist, color='blue', linewidth=2)
        axes[1, 0].set_title('픽셀 강도 히스토그램', fontsize=14)
        axes[1, 0].set_xlabel('픽셀 강도')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].grid(True, alpha=0.3)

        small_gray = cv2.resize(gray, (50, 50))
        x = np.arange(0, small_gray.shape[1])
        y = np.arange(0, small_gray.shape[0])
        X, Y = np.meshgrid(x, y)

        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        ax_3d.plot_surface(X, Y, small_gray, cmap='coolwarm', alpha=0.8)
        ax_3d.set_title('3D 픽셀 강도 표면', fontsize=14)
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('강도')

        plt.tight_layout()
        return fig

    def geometric_shape_graph(self):
        """기하학적 형태 분석 그래프"""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.imshow(self.original_image)
        ax1.set_title('원본 이미지', fontsize=14)
        ax1.axis('off')

        ax2.set_xlim(0, self.original_image.shape[1])
        ax2.set_ylim(self.original_image.shape[0], 0)

        shape_colors = {'삼각형': 'red', '사각형': 'blue', '원형': 'green', '기타': 'orange'}

        for contour in contours:
            if cv2.contourArea(contour) > 100:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 3:
                    color = shape_colors['삼각형']
                elif len(approx) == 4:
                    color = shape_colors['사각형']
                elif len(approx) > 8:
                    color = shape_colors['원형']
                else:
                    color = shape_colors['기타']

                x_coords = contour[:, 0, 0]
                y_coords = contour[:, 0, 1]
                ax2.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)

        ax2.set_title('기하학적 형태 분석', fontsize=14)
        ax2.set_aspect('equal')

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=color, lw=2, label=shape)
                           for shape, color in shape_colors.items()]
        ax2.legend(handles=legend_elements)

        plt.tight_layout()
        return fig


def main():
    """메인 실행 함수"""
    print("GUI 이미지 분석기를 시작합니다...")
    print("필요한 라이브러리: opencv-python, matplotlib, scikit-learn, scipy, pillow")
    print("-" * 60)

    try:
        root = tk.Tk()
        app = ImageAnalyzerGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"프로그램 실행 오류: {e}")
        messagebox.showerror("오류", f"프로그램 실행 실패: {str(e)}")


if __name__ == "__main__":
    main()