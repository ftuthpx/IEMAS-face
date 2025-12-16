#!/usr/bin/env python3
"""
動画クリッピングツール
複数カメラの映像が1つの画面に統合された動画ファイルから、
特定のカメラ映像部分のみを切り出すコマンドラインツール
"""

import os
import sys
import glob
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tqdm import tqdm


class RegionSelector:
    """マウスドラッグで矩形範囲を選択するクラス"""

    def __init__(self, image: np.ndarray, window_name: str = "Select Region"):
        self.original_image = image.copy()
        self.image = image.copy()
        self.window_name = window_name
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.region_selected = False
        self.confirmed = False

    def mouse_callback(self, event, x, y, flags, param):
        """マウスイベントのコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            self.region_selected = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.image = self.original_image.copy()
                cv2.rectangle(self.image, self.start_point, self.end_point, (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self.image = self.original_image.copy()
            cv2.rectangle(self.image, self.start_point, self.end_point, (0, 255, 0), 2)
            self.region_selected = True

    def get_region(self) -> tuple:
        """選択された領域を取得 (x, y, width, height)"""
        if not self.region_selected or self.start_point is None or self.end_point is None:
            return None

        x1, y1 = self.start_point
        x2, y2 = self.end_point

        # 座標を正規化（左上が小さい値になるように）
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        if width == 0 or height == 0:
            return None

        return (x, y, width, height)

    def select(self) -> tuple:
        """
        範囲選択を実行
        Returns: (x, y, width, height) または None（キャンセル時）
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.original_image.shape[1], self.original_image.shape[0])
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n--- 範囲選択 ---")
        print("マウスドラッグで矩形範囲を選択してください")
        print("Enter または Space: 確定 / Esc: やり直し / q: 終了")
        print("-" * 20)

        while True:
            cv2.imshow(self.window_name, self.image)
            # macOSでの入力メソッド問題を回避するため、より長いwait時間を使用
            key = cv2.waitKey(50) & 0xFF

            # キー入力がない場合はスキップ
            if key == 255:
                continue

            # Enter (13), LF (10), またはSpace (32) で確定
            # macOSの日本語入力などでEnterが検出されにくい場合があるためSpaceも対応
            if key in (13, 10, 32):
                if self.region_selected:
                    region = self.get_region()
                    if region:
                        self.confirmed = True
                        cv2.destroyAllWindows()
                        return region
                    else:
                        print("有効な範囲が選択されていません。再度選択してください。")
                else:
                    print("範囲が選択されていません。マウスドラッグで範囲を選択してください。")

            elif key == 27:  # Esc
                self.image = self.original_image.copy()
                self.start_point = None
                self.end_point = None
                self.region_selected = False
                print("選択をリセットしました。再度選択してください。")

            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

        return None


def select_folder(title: str) -> str:
    """GUIでフォルダを選択"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


def get_middle_frame(video_path: str) -> np.ndarray:
    """動画の中間フレームを取得"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画ファイルを開けませんでした: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_idx = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"フレームを読み込めませんでした: {video_path}")

    return frame


def get_mp4_files(folder: str) -> list:
    """フォルダ内のMP4ファイル一覧を取得（サブフォルダは除外）"""
    mp4_files = []
    for file in os.listdir(folder):
        if file.lower().endswith('.mp4'):
            full_path = os.path.join(folder, file)
            if os.path.isfile(full_path):
                mp4_files.append(full_path)
    return sorted(mp4_files)


def clip_video(input_path: str, output_path: str, region: tuple) -> bool:
    """
    動画をクリッピング

    Args:
        input_path: 入力動画ファイルパス
        output_path: 出力動画ファイルパス
        region: (x, y, width, height)

    Returns:
        成功したかどうか
    """
    x, y, width, height = region

    # ffmpegコマンドを構築
    # cropフィルターは再エンコードが必要だが、高品質設定を使用
    cmd = [
        'ffmpeg',
        '-y',  # 上書き確認なし
        '-i', input_path,
        '-vf', f'crop={width}:{height}:{x}:{y}',
        '-c:v', 'libx264',
        '-crf', '18',  # 高品質設定（0が無劣化、18は視覚的にほぼ無劣化）
        '-preset', 'medium',
        '-c:a', 'copy',  # 音声はそのままコピー
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"エラー: {e}")
        return False


def main():
    """メイン処理"""
    print("=" * 50)
    print("動画クリッピングツール")
    print("=" * 50)

    # 入力フォルダ選択
    print("\n入力フォルダを選択してください...")
    input_folder = select_folder("入力フォルダを選択")
    if not input_folder:
        print("入力フォルダが選択されませんでした。終了します。")
        sys.exit(1)
    print(f"入力フォルダ: {input_folder}")

    # 出力フォルダ選択
    print("\n出力フォルダを選択してください...")
    output_folder = select_folder("出力フォルダを選択")
    if not output_folder:
        print("出力フォルダが選択されませんでした。終了します。")
        sys.exit(1)
    print(f"出力フォルダ: {output_folder}")

    # MP4ファイル一覧を取得
    mp4_files = get_mp4_files(input_folder)
    if not mp4_files:
        print("入力フォルダにMP4ファイルが見つかりませんでした。終了します。")
        sys.exit(1)
    print(f"\n{len(mp4_files)}個のMP4ファイルが見つかりました。")

    # 最初のファイルの中間フレームを取得
    first_file = mp4_files[0]
    print(f"\n参照ファイル: {os.path.basename(first_file)}")
    print("中間フレームを読み込んでいます...")

    try:
        frame = get_middle_frame(first_file)
    except RuntimeError as e:
        print(f"エラー: {e}")
        sys.exit(1)

    # 範囲選択
    selector = RegionSelector(frame)
    region = selector.select()

    if region is None:
        print("範囲選択がキャンセルされました。終了します。")
        sys.exit(0)

    x, y, width, height = region
    print(f"\n選択された範囲: x={x}, y={y}, width={width}, height={height}")

    # 一括処理
    print(f"\n{'=' * 50}")
    print("クリッピング処理を開始します...")
    print(f"{'=' * 50}\n")

    success_count = 0
    error_files = []

    for mp4_file in tqdm(mp4_files, desc="処理中", unit="ファイル"):
        filename = os.path.basename(mp4_file)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_clipped{ext}"
        output_path = os.path.join(output_folder, output_filename)

        tqdm.write(f"処理中: {filename}")

        if clip_video(mp4_file, output_path, region):
            success_count += 1
        else:
            error_files.append(filename)
            tqdm.write(f"エラー: {filename} の処理に失敗しました")
            print(f"\n処理を停止しました。エラーファイル: {filename}")
            break

    # 結果表示
    print(f"\n{'=' * 50}")
    print("処理完了")
    print(f"{'=' * 50}")
    print(f"成功: {success_count}/{len(mp4_files)} ファイル")

    if error_files:
        print(f"\nエラーが発生したファイル:")
        for f in error_files:
            print(f"  - {f}")
    else:
        print("\nすべてのファイルが正常に処理されました。")

    print(f"\n出力先: {output_folder}")


if __name__ == "__main__":
    main()
