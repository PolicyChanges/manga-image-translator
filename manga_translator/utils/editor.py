import os
import platform
import subprocess
import threading
import queue
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import freetype
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import colorchooser, filedialog, ttk

from .textblock import TextBlock
from ..rendering import text_render
from ..utils import BASE_PATH, rect_distance

image_stream_queue: queue.Queue = queue.Queue()
shutdown_event: threading.Event = threading.Event()
    
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PUNSET_RIGHT_ENG = {'.', '?', '!', ':', ';', ')', '}', '"'}

class Textline:
    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0, spacing: int = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)
        self.num_words = 1 if text else 0
        self.spacing = 0
        self.add_spacing(spacing)

    def append_right(self, word: str, w_len: int, delimiter: str = ''):
        self.text = f"{self.text}{delimiter}{word}" if self.text else word
        if word:
            self.num_words += 1
        self.length += w_len

    def append_left(self, word: str, w_len: int, delimiter: str = ''):
        self.text = f"{word}{delimiter}{self.text}" if self.text else word
        if word:
            self.num_words += 1
        self.length += w_len

    def add_spacing(self, spacing: int):
        self.spacing = spacing
        self.pos_x -= spacing
        self.length += 2 * spacing

    def strip_spacing(self):
        self.length -= self.spacing * 2
        self.pos_x += self.spacing
        self.spacing = 0

def render_lines(
    textlines: List[Textline],
    canvas_h: int,
    canvas_w: int,
    font_size: int,
    stroke_width: int,
    line_spacing: float = 0.01,
    fg: Tuple[int, int, int] = (0, 0, 0),
    bg: Tuple[int, int, int] = (255, 255, 255)
) -> Optional[Image.Image]:
    if not textlines:
        return None

    bg_size = stroke_width
    spacing_y = int(font_size * (line_spacing or 0.01))

    max_len = max(line.length for line in textlines)
    calc_w = max_len + (font_size + bg_size) * 2
    calc_h = font_size * len(textlines) + spacing_y * (len(textlines) - 1) + (font_size + bg_size) * 2

    canvas_text = np.zeros((calc_h, calc_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    pen_orig = [font_size + bg_size, font_size + bg_size]

    for line in textlines:
        pen_line = pen_orig.copy()
        pen_line[0] += line.pos_x
        for char in line.text:
            offset_x = text_render.put_char_horizontal(font_size, char, pen_line, canvas_text, canvas_border, border_size=bg_size)
            pen_line[0] += offset_x
        pen_orig[1] += spacing_y + font_size

    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = text_render.add_color(canvas_text, fg, canvas_border, bg)

    x, y, width, height = cv2.boundingRect(canvas_border)
    if width <= 0 or height <= 0:
        return None

    return Image.fromarray(line_box[y:y + height, x:x + width])


def seg_eng(text: str) -> List[str]:
    text = text.strip().upper().replace('  ', ' ').replace(' .', '.').replace('\n', ' ')
    processed_text = []

    text_len = len(text)
    for i, char in enumerate(text):
        if char in PUNSET_RIGHT_ENG and i < text_len - 1:
            next_c = text[i + 1]
            processed_text.append(f"{char} " if next_c.isalnum() else char)
        else:
            processed_text.append(char)

    word_list = ''.join(processed_text).split(' ')
    word_num = len(word_list)
    if word_num <= 1:
        return word_list

    words = []
    skip_next = False
    for i, word in enumerate(word_list):
        if skip_next:
            skip_next = False
            continue

        if len(word) < 3:
            append_left, append_right = False, False
            len_word = len(word)
            len_next = len(word_list[i + 1]) if i < word_num - 1 else -1
            len_prev = len(words[-1]) if words else -1

            cond_next = (len_word == 2 and len_next <= 4) or len_word == 1
            cond_prev = (len_word == 2 and len_prev <= 4) or len_word == 1

            if len_next > 0 and len_prev > 0:
                append_right = cond_next if len_next < len_prev else False
                append_left = cond_prev if not append_right else False
            elif len_next > 0:
                append_right = cond_next
            elif len_prev > 0:
                append_left = cond_prev

            if append_left and words:
                words[-1] = f"{words[-1]} {word}"
            elif append_right:
                words.append(f"{word} {word_list[i + 1]}")
                skip_next = True
            else:
                words.append(word)
            continue
        words.append(word)

    return words


def layout_lines_aligncenter(
    mask: np.ndarray,
    words: List[str],
    word_lengths: List[int],
    delimiter_len: int,
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    max_central_width: float = np.inf
) -> List[Textline]:
    m = cv2.moments(mask)
    if m['m00'] == 0:
        centroid_x, centroid_y = mask.shape[1] // 2, mask.shape[0] // 2
    else:
        centroid_y = int(m['m01'] / m['m00'])
        centroid_x = int(m['m10'] / m['m00'])

    mask = 255 - mask
    bh, bw = mask.shape[:2]
    num_words = len(words)
    wlst_left, wlst_right = [], []
    len_left, len_right = [], []
    sum_left, sum_right = 0, 0

    if num_words > 1:
        wl_array = np.array(word_lengths, dtype=np.float64)
        wl_cumsums = np.cumsum(wl_array)
        wl_cumsums = wl_cumsums - wl_cumsums[-1] / 2 - wl_array / 2
        central_index = int(np.argmin(np.abs(wl_cumsums)))

        if central_index > 0:
            wlst_left = words[:central_index]
            len_left = word_lengths[:central_index]
            sum_left = sum(len_left)
        if central_index < num_words - 1:
            wlst_right = words[central_index + 1:]
            len_right = word_lengths[central_index + 1:]
            sum_right = sum(len_right)
    else:
        central_index = 0

    pos_y = centroid_y - line_height // 2
    pos_x = centroid_x - word_lengths[central_index] // 2
    central_line = Textline(words[central_index], pos_x, pos_y, word_lengths[central_index], spacing)
    line_bottom = pos_y + line_height

    while sum_left > 0 or sum_right > 0:
        left_valid, right_valid = False, False
        new_x_l, new_x_r = 0, 0

        if sum_left > 0:
            new_len_l = central_line.length + len_left[-1] + delimiter_len
            new_x_l = centroid_x - new_len_l // 2
            new_r_l = new_x_l + new_len_l
            if 0 < new_x_l and new_r_l < bw:
                if mask[pos_y:line_bottom, new_x_l].sum() == 0 and mask[pos_y:line_bottom, new_r_l].sum() == 0:
                    left_valid = True

        if sum_right > 0:
            new_len_r = central_line.length + len_right[0] + delimiter_len
            new_x_r = centroid_x - new_len_r // 2
            new_r_r = new_x_r + new_len_r
            if 0 < new_x_r and new_r_r < bw:
                if mask[pos_y:line_bottom, new_x_r].sum() == 0 and mask[pos_y:line_bottom, new_r_r].sum() == 0:
                    right_valid = True

        insert_left = (sum_left > sum_right) if (left_valid and right_valid) else left_valid
        if not (left_valid or right_valid):
            break

        if insert_left:
            central_line.append_left(wlst_left.pop(-1), len_left[-1] + delimiter_len, delimiter)
            sum_left -= len_left.pop(-1)
            central_line.pos_x = new_x_l
        else:
            central_line.append_right(wlst_right.pop(0), len_right[0] + delimiter_len, delimiter)
            sum_right -= len_right.pop(0)
            central_line.pos_x = new_x_r

        if central_line.length > max_central_width:
            break

    central_line.strip_spacing()
    lines = [central_line]

    # Layout bottom half
    if sum_right > 0:
        w, wl = wlst_right.pop(0), len_right.pop(0)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y + line_height // 2
        line_bottom = pos_y + line_height
        line = Textline(w, pos_x, pos_y, wl, spacing)
        lines.append(line)
        sum_right -= wl

        while sum_right > 0:
            w, wl = wlst_right.pop(0), len_right.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len

            valid = (0 < new_x and right_x < bw) and (
                mask[pos_y:line_bottom, new_x].sum() == 0 and mask[pos_y:line_bottom, right_x].sum() == 0
            )

            if valid:
                line.append_right(w, wl + delimiter_len, delimiter)
                line.pos_x = new_x
                if new_len > max_central_width:
                    line.strip_spacing()
                    break
            else:
                line.strip_spacing()
                pos_x = centroid_x - wl // 2
                pos_y = line_bottom
                line_bottom += line_height
                line = Textline(w, pos_x, pos_y, wl, spacing)
                lines.append(line)

    # Layout top half
    if sum_left > 0:
        w, wl = wlst_left.pop(-1), len_left.pop(-1)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y - line_height * 2
        line_bottom = pos_y + line_height
        line = Textline(w, pos_x, pos_y, wl, spacing)
        lines.insert(0, line)
        sum_left -= wl

        while sum_left > 0:
            w, wl = wlst_left.pop(-1), len_left.pop(-1)
            sum_left -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len

            valid = (0 < new_x and right_x < bw) and (
                mask[pos_y:line_bottom, new_x].sum() == 0 and mask[pos_y:line_bottom, right_x].sum() == 0
            )

            if valid:
                line.append_left(w, wl + delimiter_len, delimiter)
                line.pos_x = new_x
                if new_len > max_central_width:
                    line.strip_spacing()
                    break
            else:
                line.strip_spacing()
                pos_x = centroid_x - wl // 2
                pos_y -= line_height
                line_bottom = pos_y + line_height
                line = Textline(w, pos_x, pos_y, wl, spacing)
                lines.insert(0, line)

    return lines


def render_textblock_list_eng_rendered(
    img: np.ndarray,
    text_regions: List[TextBlock],
    rendered_text_list: Optional[List] = None,
    rendered_text_coords: Optional[List] = None,
    delimiter: str = ' ',
    line_spacing: float = 0.01,
    stroke_width: float = 0.1,
    ballonarea_thresh: float = 2.0,
    downscale_constraint: float = 0.7,
    original_img: Optional[np.ndarray] = None,
    disable_font_border: bool = False
    #downscale_font_size: float = 1.0
) -> Tuple[List, List]:
    if rendered_text_list is None:
        rendered_text_list = []
    if rendered_text_coords is None:
        rendered_text_coords = []
        
    #for text_region in text_regions:
    #    text_region.font_size=int(text_region.font_size*downscale_font_size)
        
    def calculate_font_values(f_size: int, words_in: List[str]):
        f_size = int(f_size)
        sw = int(f_size * stroke_width)
        line_height = int(f_size * 0.8)
        delim_glyph = text_render.get_char_glyph(delimiter, f_size, 0)
        delim_len = delim_glyph.advance.x >> 6
        b_length = -1
        w_lengths = []
        for word in words_in:
            w_len = sum((text_render.get_char_glyph(c, f_size, 0).metrics.horiAdvance >> 6) for c in word)
            w_lengths.append(w_len)
            if w_len > b_length:
                b_length = w_len
        return f_size, sw, line_height, delim_len, b_length, w_lengths

    def update_enlarged_xyxy(reg):
        reg.enlarged_xyxy = reg.xyxy.copy()
        w_diff, h_diff = ((reg.xywh[2:] * reg.enlarge_ratio) - reg.xywh[2:].astype(np.float64)) // 2
        reg.enlarged_xyxy[0] -= w_diff
        reg.enlarged_xyxy[2] += w_diff
        reg.enlarged_xyxy[1] -= h_diff
        reg.enlarged_xyxy[3] += h_diff

    for region in text_regions:
        region.enlarge_ratio = 1
        region.enlarged_xyxy = region.xyxy.copy()
    
    for region in text_regions:
        if region.enlarge_ratio == 1:
            region.enlarge_ratio = min(max(region.xywh[2] / region.xywh[3], region.xywh[3] / region.xywh[2]) * 1.5, 3)
            update_enlarged_xyxy(region)

        for region2 in text_regions:
            if region is region2:
                continue
            if rect_distance(*region.enlarged_xyxy, *region2.enlarged_xyxy) == 0:
                d = rect_distance(*region.xyxy, *region2.xyxy)
                l1 = (region.xywh[2] + region.xywh[3]) / 2
                l2 = (region2.xywh[2] + region2.xywh[3]) / 2
                region.enlarge_ratio = d / (2 * l1) + 1
                region2.enlarge_ratio = d / (2 * l2) + 1
                update_enlarged_xyxy(region)
                update_enlarged_xyxy(region2)

    for region in text_regions:
        words = seg_eng(region.translation)
        if not words:
            continue

        font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(
            region.font_size, words
        )

        ballon_mask, xyxy = extract_ballon_region(original_img, region.xywh, enlarge_ratio=region.enlarge_ratio)
        ballon_area = (ballon_mask > 0).sum()
        rotated, rx, ry = False, 0.0, 0.0

        if abs(region.angle) > 3:
            rotated = True
            angle_rad = np.deg2rad(region.angle)
            angle_sin, angle_cos = np.sin(angle_rad), np.cos(angle_rad)
            rotated_ballon_mask = np.array(Image.fromarray(ballon_mask).rotate(region.angle, expand=True))

            deg = region.angle % 360
            if 0 < deg <= 90:
                ry = abs(ballon_mask.shape[1] * angle_sin)
            elif 90 < deg <= 180:
                rx = abs(ballon_mask.shape[1] * angle_cos)
                ry = rotated_ballon_mask.shape[0]
            elif 180 < deg <= 270:
                ry = abs(ballon_mask.shape[0] * angle_cos)
                rx = rotated_ballon_mask.shape[1]
            else:
                rx = abs(ballon_mask.shape[0] * angle_sin)
            ballon_mask = rotated_ballon_mask

        line_width = sum(word_lengths) + delimiter_len * (len(word_lengths) - 1)
        region_area = line_width * line_height + delimiter_len * (len(words) - 1) * line_height
        area_ratio = (ballon_area / region_area) if region_area > 0 else 1.0
        resize_ratio = 1.0

        if area_ratio < ballonarea_thresh and area_ratio > 0:
            scale_factor = ballonarea_thresh / area_ratio
            resize_ratio = min(np.sqrt(scale_factor), (1 / downscale_constraint) ** 2)
            rx *= resize_ratio
            ry *= resize_ratio
            new_shape = (int(resize_ratio * ballon_mask.shape[1]), int(resize_ratio * ballon_mask.shape[0]))
            ballon_mask = cv2.resize(ballon_mask, new_shape)

        nonzero_pts = cv2.findNonZero(ballon_mask)
        if nonzero_pts is None:
            continue
        region_x, region_y, region_w, region_h = cv2.boundingRect(nonzero_pts)

        base_word_idx = max(enumerate(word_lengths), key=lambda x: x[1])[0]
        base_length_word = words[base_word_idx]
        if not base_length_word:
            continue

        lines_needed = len(region.translation) / len(base_length_word)
        lines_available = abs(xyxy[3] - xyxy[1]) // line_height + 1
        font_size_multiplier = max(
            min(region_w / (base_length + 2 * sw), lines_available / lines_needed),
            downscale_constraint
        )

        if font_size_multiplier < 1:
            font_size = int(font_size * font_size_multiplier)
            font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(font_size, words)

        textlines = layout_lines_aligncenter(
            ballon_mask, words, word_lengths, delimiter_len, line_height, delimiter=delimiter
        )
        if not textlines:
            continue

        line_cy = np.mean([line.pos_y for line in textlines]) + line_height / 2
        region_cy = region_y + region_h / 2
        y_offset = int(round(np.clip(region_cy - line_cy, -line_height, line_height)))

        lines_x1 = np.array([line.pos_x for line in textlines])
        lines_x2 = np.array([max(line.pos_x, 0) + line.length for line in textlines])
        canvas_x1, canvas_x2 = lines_x1.min() - sw, lines_x2.max() + sw
        canvas_y1, canvas_y2 = textlines[0].pos_y - sw, textlines[-1].pos_y + line_height + sw
        canvas_h = int(canvas_y2 - canvas_y1)
        canvas_w = int(canvas_x2 - canvas_x1)

        for line in textlines:
            line.pos_x -= canvas_x1
            line.pos_y -= canvas_y1

        region_font_color, region_stroke_color = region.get_font_colors()
        if getattr(region, 'fg_color', None) is not None:
            region_font_color = region.fg_color
        elif getattr(region, 'font_color', None) is not None:
            region_font_color = region.font_color

        if getattr(region, 'bg_color', None) is not None:
            region_stroke_color = region.bg_color
        elif getattr(region, 'stroke_color', None) is not None:
            region_stroke_color = region.stroke_color

        textlines_image = render_lines(
            textlines, canvas_h, canvas_w, font_size, sw, line_spacing, region_font_color, region_stroke_color
        )
        if textlines_image is None:
            continue

        rel_cx = ((canvas_x1 + canvas_x2) / 2 - rx) / resize_ratio
        rel_cy = ((canvas_y1 + canvas_y2) / 2 - ry + y_offset) / resize_ratio

        if rotated:
            rcx = rel_cx * angle_cos - rel_cy * angle_sin
            rcy = rel_cx * angle_sin + rel_cy * angle_cos
            rel_cx, rel_cy = rcx, rcy
            textlines_image = textlines_image.rotate(-region.angle, expand=True, resample=Image.Resampling.BILINEAR)
            bbox = textlines_image.getbbox()
            if bbox:
                textlines_image = textlines_image.crop(bbox)

        abs_x = int(rel_cx + xyxy[0] - textlines_image.width / 2)
        abs_y = int(rel_cy + xyxy[1] - textlines_image.height / 2)

        rendered_text_list.append(textlines_image)
        rendered_text_coords.append((abs_x, abs_y))

    return rendered_text_list, rendered_text_coords


class LiveEditor(tk.Frame):
    def __init__(self, master, initial_image_dict=None, **kwargs):
        super().__init__(master, **kwargs)

        self.active_layer_id = None
        self.active_layer_index = None
        self.drag_data = {"x": 0, "y": 0}
        self.current_bg_pil = None
        self.tk_bg_image = None
        self.thumbnail_refs = []
        self.layer_image_refs = {}
        self.bg_width = 0
        self.bg_height = 0

        self.gallery_images = []
        self.current_img_index = -1
        self._tk_overlay_cache = []

        # Closable Property Box widget references
        self.property_box: Optional[tk.Toplevel] = None
        self.layer_font_size_display: Optional[tk.Entry] = None
        self.fg_color_swatch: Optional[tk.Label] = None
        self.fg_color_pick_btn: Optional[tk.Button] = None
        self.bg_color_swatch: Optional[tk.Label] = None
        self.bg_color_pick_btn: Optional[tk.Button] = None
        self.layer_text_display: Optional[tk.Entry] = None
        self.render_btn: Optional[tk.Button] = None

        self.current_fg_color: Tuple[int, int, int] = (0, 0, 0)
        self.current_bg_color: Tuple[int, int, int] = (255, 255, 255)

        self._build_top_controls()
        self._build_main_workspace()

        if initial_image_dict:
            self.append_image(initial_image_dict)
        else:
            self._show_placeholder_text()

        self.bind_all("<Left>", self._on_arrow_left_press)
        self.bind_all("<Right>", self._on_arrow_right_press)

    def _build_top_controls(self):
        self.input_frame = tk.Frame(self, bg="#f0f0f0")
        self.input_frame.pack(side="top", fill="x", pady=5)

        # Packed from the right in order:
        # [Export Image] [Export All] [Autosize] [Exit]

        # 1. Exit Button (rightmost)
        self.exit_btn = tk.Button(
            self.input_frame,
            text="Exit",
            font=("Arial", 10, "bold"),
            bg="#d9d9d9",
            fg="black",
            activebackground="#bfbfbf",
            activeforeground="black",
            relief="raised",
            bd=2,
            padx=10,
            pady=2,
            command=self.exit_program
        )
        self.exit_btn.pack(side="right", padx=(5, 15))

        # 2. Autosize to Window (to the left of Exit)
        self.autosize_enabled = tk.BooleanVar(value=False)
        self.autosize_toggle = tk.Checkbutton(
            self.input_frame,
            text="🔄 Autosize to Window",
            font=("Arial", 11),
            bg="#f0f0f0",
            fg="black",
            activeforeground="black",
            selectcolor="#ffffff",
            anchor="w",
            justify="left",
            variable=self.autosize_enabled,
            command=self._on_autosize_toggle_click
        )
        self.autosize_toggle.pack(side="right", padx=10)

        # 3. Export All Button (between Export Image and Autosize)
        self.export_all_btn = tk.Button(
            self.input_frame,
            text="Export All",
            font=("Arial", 10, "bold"),
            bg="#d9d9d9",
            fg="black",
            activebackground="#bfbfbf",
            activeforeground="black",
            disabledforeground="#808080",
            relief="raised",
            bd=2,
            padx=10,
            pady=2,
            command=self.export_all_images
        )
        self.export_all_btn.pack(side="right", padx=5)

        # 4. Export Image Button (to the left of Export All)
        self.export_img_btn = tk.Button(
            self.input_frame,
            text="Export Image",
            font=("Arial", 10, "bold"),
            bg="#d9d9d9",
            fg="black",
            activebackground="#bfbfbf",
            activeforeground="black",
            disabledforeground="#808080",
            relief="raised",
            bd=2,
            padx=10,
            pady=2,
            command=self.export_current_image
        )
        self.export_img_btn.pack(side="right", padx=5)

    def exit_program(self):
        """Cleanly releases graphics/memory resources and terminates the program entirely."""
        try:
            self.on_close_cleanup()
        except Exception:
            pass
        finally:
            os._exit(0)

    def _build_main_workspace(self):
        self.workspace = tk.Frame(self, bg="#dcdcdc")
        self.workspace.pack(side="top", fill="both", expand=True)

        self.sidebar_container = tk.Frame(self.workspace, width=150, bg="#e0e0e0", bd=1, relief="sunken")
        self.sidebar_container.pack(side="left", fill="y")

        self.gallery_canvas = tk.Canvas(self.sidebar_container, width=130, bg="#e0e0e0", bd=0, highlightthickness=0)
        self.gallery_scrollbar = ttk.Scrollbar(self.sidebar_container, orient="vertical", command=self.gallery_canvas.yview)
        self.scrollable_gallery_frame = tk.Frame(self.gallery_canvas, bg="#e0e0e0")

        self.scrollable_gallery_frame.bind(
            "<Configure>",
            lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
        )
        self.gallery_canvas.create_window((0, 0), window=self.scrollable_gallery_frame, anchor="nw")
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scrollbar.set)

        self.gallery_canvas.pack(side="left", fill="both", expand=True)
        self.gallery_scrollbar.pack(side="right", fill="y")

        self.canvas_container = tk.Frame(self.workspace, bg="gray")
        self.canvas_container.pack(side="right", fill="both", expand=True)

        self.v_scrollbar = ttk.Scrollbar(self.canvas_container, orient="vertical")
        self.h_scrollbar = ttk.Scrollbar(self.canvas_container, orient="horizontal")

        self.canvas = tk.Canvas(
            self.canvas_container, bd=0, highlightthickness=0, bg="gray",
            xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set
        )

        self.v_scrollbar.config(command=self.canvas.yview)
        self.h_scrollbar.config(command=self.canvas.xview)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas_container.rowconfigure(0, weight=1)
        self.canvas_container.columnconfigure(0, weight=1)

        self.gallery_canvas.bind("<MouseWheel>", self._on_sidebar_mousewheel)
        self.canvas.bind("<MouseWheel>", self._on_main_canvas_mousewheel)
        self.gallery_canvas.bind("<Button-4>", self._on_sidebar_mousewheel)
        self.gallery_canvas.bind("<Button-5>", self._on_sidebar_mousewheel)
        self.canvas.bind("<Button-4>", self._on_main_canvas_mousewheel)
        self.canvas.bind("<Button-5>", self._on_main_canvas_mousewheel)

        self.canvas_container.bind("<Configure>", self._on_window_resize_event)

    def _show_placeholder_text(self):
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, 500, 400))
        self.canvas.create_text(
            250, 200,
            text="No images loaded.\n\nPlease append images to start editing.",
            fill="white", font=("Arial", 12, "italic"), justify="center", tags="placeholder"
        )

    def append_image(self, image_dict: dict, path: str = ""):
        for r in image_dict['text_regions']:
            print(f"FONT_SIZE: {r.font_size}")
        rendered_text_list, rendered_text_coords = render_textblock_list_eng_rendered(
            img=image_dict['img_inpainted'],
            text_regions=image_dict['text_regions'],
            rendered_text_list=[],
            rendered_text_coords=[],
            original_img=image_dict['img_rgb'],
        )

        rendered_text_translated = [
            " ".join(seg_eng(r.translation)) for r in image_dict['text_regions']
        ]
        gallery_image = Image.fromarray(image_dict['img_inpainted'])

        if not isinstance(gallery_image, Image.Image):
            return

        threading.Thread(
            target=self._async_gallery_worker,
            args=(
                gallery_image,
                rendered_text_list,
                rendered_text_coords,
                rendered_text_translated,
                image_dict,
                path,
            ),
            daemon=True,
        ).start()

    def _async_gallery_worker(
        self,
        gallery_image,
        rendered_text_list,
        rendered_text_coords,
        rendered_text_translated,
        image_dict,
        export_path: str = "",
    ):
        try:
            thumb_pil = Image.fromarray(image_dict['img_rgb']).copy()
            thumb_pil.thumbnail((110, 80))
            thumb_tk = ImageTk.PhotoImage(thumb_pil)

            dest_path = export_path or image_dict.get('export_path', '') or image_dict.get('dest', '')

            data_bundle = {
                "bg": gallery_image,
                "layers": rendered_text_list,
                "coords": rendered_text_coords,
                "text_strings": rendered_text_translated,
                "img_rgb": image_dict['img_rgb'],
                "img_inpainted": image_dict['img_inpainted'],
                "text_regions": image_dict['text_regions'],
                "export_path": dest_path,
            }
            self.master.after(0, self._safe_append_thumbnail, data_bundle, thumb_tk)
        except Exception as e:
            print(f"Background processing error: {e}")

    # -------------------------------------------------------------
    # Image Composition and Export Pipeline
    # -------------------------------------------------------------
    def _composite_bundle(self, bundle: dict) -> Optional[Image.Image]:
        bg_pil = bundle.get("bg")
        if bg_pil is None:
            return None

        comp = bg_pil.convert("RGBA").copy()
        layers = bundle.get("layers", [])
        coords = bundle.get("coords", [])

        for i, layer in enumerate(layers):
            if not isinstance(layer, Image.Image):
                continue
            x, y = coords[i] if i < len(coords) else (50, 50)
            layer_rgba = layer.convert("RGBA")
            comp.paste(layer_rgba, (int(x), int(y)), mask=layer_rgba)

        return comp

    def _save_composition(self, bundle: dict, export_path: str) -> bool:
        if not export_path:
            return False

        comp = self._composite_bundle(bundle)
        if comp is None:
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
            ext = os.path.splitext(export_path)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                comp.convert("RGB").save(export_path, quality=95)
            else:
                comp.save(export_path)
            print(f"Exported composition to: {export_path}")
            return True
        except Exception as e:
            print(f"Error exporting image to {export_path}: {e}")
            return False

    def export_current_image(self):
        if self.autosize_enabled.get():
            return
        if self.current_img_index == -1 or not self.gallery_images:
            return

        bundle = self.gallery_images[self.current_img_index]
        export_path = bundle.get("export_path", "")
        if not export_path:
            export_path = filedialog.asksaveasfilename(
                title="Save Composition",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg"), ("All Files", "*.*")]
            )
            if export_path:
                bundle["export_path"] = export_path

        if export_path:
            self._save_composition(bundle, export_path)

    def export_all_images(self):
        if self.autosize_enabled.get():
            return
        if not self.gallery_images:
            return

        success_count = 0
        for i, bundle in enumerate(self.gallery_images):
            export_path = bundle.get("export_path", "")
            if not export_path:
                print(f"Skipping gallery item {i + 1}: No destination path set.")
                continue
            if self._save_composition(bundle, export_path):
                success_count += 1

        print(f"Batch export completed: {success_count}/{len(self.gallery_images)} image(s) saved.")

    # -------------------------------------------------------------
    # Closable Property Box Management
    # -------------------------------------------------------------
    def open_or_update_property_box(self, layer_index: int, event=None):
        active_bundle = self.gallery_images[self.current_img_index] if self.current_img_index != -1 else {}
        text_strings = active_bundle.get("text_strings", [])
        text_regions = active_bundle.get("text_regions", [])

        display_text = text_strings[layer_index] if layer_index < len(text_strings) else ""
        region = text_regions[layer_index] if layer_index < len(text_regions) else None
        font_size = str(getattr(region, 'font_size', '')) if region else ""

        current_fg = getattr(region, 'fg_color', None) or getattr(region, 'font_color', None)
        current_bg = getattr(region, 'bg_color', None) or getattr(region, 'stroke_color', None)

        if (current_fg is None or current_bg is None) and hasattr(region, 'get_font_colors'):
            try:
                def_fg, def_bg = region.get_font_colors()
                if current_fg is None:
                    current_fg = def_fg
                if current_bg is None:
                    current_bg = def_bg
            except Exception:
                pass

        if current_fg is None:
            current_fg = (0, 0, 0)
        else:
            current_fg = tuple(int(c) for c in current_fg[:3])

        if current_bg is None:
            current_bg = (255, 255, 255)
        else:
            current_bg = tuple(int(c) for c in current_bg[:3])

        self.current_fg_color = current_fg
        self.current_bg_color = current_bg

        fg_hex = f"#{current_fg[0]:02x}{current_fg[1]:02x}{current_fg[2]:02x}"
        bg_hex = f"#{current_bg[0]:02x}{current_bg[1]:02x}{current_bg[2]:02x}"

        if self.property_box is None or not self.property_box.winfo_exists():
            self.property_box = tk.Toplevel(self)
            self.property_box.transient(self.winfo_toplevel())
            self.property_box.resizable(False, False)
            self.property_box.protocol("WM_DELETE_WINDOW", self.close_property_box)
            self.property_box.config(bg="#f4f4f4")

            if event is not None:
                x = event.x_root + 15
                y = event.y_root + 15
                self.property_box.geometry(f"+{x}+{y}")

            content = tk.Frame(self.property_box, bg="#f4f4f4", padx=14, pady=12)
            content.pack(fill="both", expand=True)

            # 1. Font Size
            size_frame = tk.Frame(content, bg="#f4f4f4")
            size_frame.pack(fill="x", pady=(0, 6))
            tk.Label(size_frame, text="Font Size:", font=("Arial", 10, "bold"), bg="#f4f4f4", width=12, anchor="w").pack(side="left")
            self.layer_font_size_display = tk.Entry(size_frame, font=("Arial", 10), width=10, bg="#ffffff", fg="#000000")
            self.layer_font_size_display.pack(side="left", padx=5)
            self.layer_font_size_display.bind("<KeyRelease>", self._on_layer_font_size_modified)

            # 2. Font Color (FG)
            fg_frame = tk.Frame(content, bg="#f4f4f4")
            fg_frame.pack(fill="x", pady=(0, 6))
            tk.Label(fg_frame, text="Font Color:", font=("Arial", 10, "bold"), bg="#f4f4f4", width=12, anchor="w").pack(side="left")

            self.fg_color_swatch = tk.Label(fg_frame, width=5, height=1, relief="solid", bd=1, bg=fg_hex)
            self.fg_color_swatch.pack(side="left", padx=(5, 6))

            self.fg_color_pick_btn = tk.Button(
                fg_frame,
                text="🎨 Choose...",
                font=("Arial", 9),
                command=self._on_choose_fg_color
            )
            self.fg_color_pick_btn.pack(side="left")

            # 3. Background Color (BG / Stroke)
            bg_frame = tk.Frame(content, bg="#f4f4f4")
            bg_frame.pack(fill="x", pady=(0, 6))
            tk.Label(bg_frame, text="BG Color:", font=("Arial", 10, "bold"), bg="#f4f4f4", width=12, anchor="w").pack(side="left")

            self.bg_color_swatch = tk.Label(bg_frame, width=5, height=1, relief="solid", bd=1, bg=bg_hex)
            self.bg_color_swatch.pack(side="left", padx=(5, 6))

            self.bg_color_pick_btn = tk.Button(
                bg_frame,
                text="🎨 Choose...",
                font=("Arial", 9),
                command=self._on_choose_bg_color
            )
            self.bg_color_pick_btn.pack(side="left")

            # 4. Layer Text
            text_frame = tk.Frame(content, bg="#f4f4f4")
            text_frame.pack(fill="x", pady=(0, 10))
            tk.Label(text_frame, text="Layer Text:", font=("Arial", 10, "bold"), bg="#f4f4f4", width=12, anchor="w").pack(side="left")
            self.layer_text_display = tk.Entry(text_frame, font=("Arial", 10), width=32, bg="#ffffff", fg="#000000")
            self.layer_text_display.pack(side="left", padx=5, fill="x", expand=True)
            self.layer_text_display.bind("<KeyRelease>", self._on_layer_text_modified)

            # 5. Buttons (Render and Close)
            btn_frame = tk.Frame(content, bg="#f4f4f4")
            btn_frame.pack(fill="x", pady=(4, 0))

            self.render_btn = tk.Button(
                btn_frame,
                text="Render Text",
                font=("Arial", 10, "bold"),
                bg="#2e7d32",
                fg="#ffffff",
                activebackground="#1b5e20",
                activeforeground="#ffffff",
                padx=12,
                pady=4,
                relief="raised",
                bd=2,
                cursor="hand2",
                command=self._on_render_text_click
            )
            self.render_btn.pack(side="left", padx=(0, 8))

            close_btn = tk.Button(
                btn_frame,
                text="Close",
                font=("Arial", 10),
                bg="#e0e0e0",
                padx=8,
                pady=4,
                command=self.close_property_box
            )
            close_btn.pack(side="right")

        self.property_box.title(f"Layer {layer_index + 1} Properties")

        self.layer_text_display.config(state="normal")
        self.layer_text_display.delete(0, tk.END)
        self.layer_text_display.insert(0, display_text)

        self.layer_font_size_display.config(state="normal")
        self.layer_font_size_display.delete(0, tk.END)
        self.layer_font_size_display.insert(0, font_size)

        self.fg_color_swatch.config(bg=fg_hex)
        self.bg_color_swatch.config(bg=bg_hex)

        if self.autosize_enabled.get():
            self.layer_text_display.config(state="disabled")
            self.layer_font_size_display.config(state="disabled")
            self.fg_color_pick_btn.config(state="disabled")
            self.bg_color_pick_btn.config(state="disabled")
        else:
            self.fg_color_pick_btn.config(state="normal")
            self.bg_color_pick_btn.config(state="normal")

        self.property_box.lift()

    def close_property_box(self):
        if self.property_box is not None:
            if self.property_box.winfo_exists():
                self.property_box.destroy()
            self.property_box = None
            self.layer_font_size_display = None
            self.fg_color_swatch = None
            self.fg_color_pick_btn = None
            self.bg_color_swatch = None
            self.bg_color_pick_btn = None
            self.layer_text_display = None
            self.render_btn = None

    def _on_choose_fg_color(self):
        if self.current_img_index == -1 or self.active_layer_index is None:
            return

        init_hex = f"#{self.current_fg_color[0]:02x}{self.current_fg_color[1]:02x}{self.current_fg_color[2]:02x}"
        parent_win = self.property_box if (self.property_box and self.property_box.winfo_exists()) else self

        chosen = colorchooser.askcolor(
            color=init_hex,
            title="Choose Font Color",
            parent=parent_win
        )

        if chosen and chosen[0] is not None:
            rgb = tuple(int(round(c)) for c in chosen[0])
            self.current_fg_color = rgb
            hex_color = chosen[1] or f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

            if self.fg_color_swatch and self.fg_color_swatch.winfo_exists():
                self.fg_color_swatch.config(bg=hex_color)

            active_bundle = self.gallery_images[self.current_img_index]
            text_regions = active_bundle.get("text_regions", [])
            if self.active_layer_index < len(text_regions):
                text_regions[self.active_layer_index].fg_color = rgb
                text_regions[self.active_layer_index].font_color = rgb

    def _on_choose_bg_color(self):
        if self.current_img_index == -1 or self.active_layer_index is None:
            return

        init_hex = f"#{self.current_bg_color[0]:02x}{self.current_bg_color[1]:02x}{self.current_bg_color[2]:02x}"
        parent_win = self.property_box if (self.property_box and self.property_box.winfo_exists()) else self

        chosen = colorchooser.askcolor(
            color=init_hex,
            title="Choose Background (Stroke) Color",
            parent=parent_win
        )

        if chosen and chosen[0] is not None:
            rgb = tuple(int(round(c)) for c in chosen[0])
            self.current_bg_color = rgb
            hex_color = chosen[1] or f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

            if self.bg_color_swatch and self.bg_color_swatch.winfo_exists():
                self.bg_color_swatch.config(bg=hex_color)

            active_bundle = self.gallery_images[self.current_img_index]
            text_regions = active_bundle.get("text_regions", [])
            if self.active_layer_index < len(text_regions):
                text_regions[self.active_layer_index].bg_color = rgb
                text_regions[self.active_layer_index].stroke_color = rgb

    def _on_layer_font_size_modified(self, event):
        if self.current_img_index == -1 or self.active_layer_index is None:
            return
        if not self.layer_font_size_display or not self.layer_font_size_display.winfo_exists():
            return

        active_bundle = self.gallery_images[self.current_img_index]
        text_regions = active_bundle.get("text_regions", [])
        val = self.layer_font_size_display.get().strip()

        if self.active_layer_index < len(text_regions):
            try:
                new_size = float(val) if '.' in val else int(val)
                text_regions[self.active_layer_index].font_size = new_size
            except ValueError:
                pass

    def _on_layer_text_modified(self, event):
        if self.current_img_index == -1 or self.active_layer_index is None:
            return
        if not self.layer_text_display or not self.layer_text_display.winfo_exists():
            return

        updated_text = self.layer_text_display.get()
        active_bundle = self.gallery_images[self.current_img_index]
        text_strings = active_bundle.setdefault("text_strings", [])

        while len(text_strings) <= self.active_layer_index:
            text_strings.append("")

        text_strings[self.active_layer_index] = updated_text

    def _on_drag_start(self, event, target_id, layer_index):
        self.open_or_update_property_box(layer_index, event)

        if self.autosize_enabled.get():
            return

        self.active_layer_id = target_id
        self.active_layer_index = layer_index
        self.canvas.itemconfig(target_id, state="normal")
        self.drag_data["x"] = self.canvas.canvasx(event.x)
        self.drag_data["y"] = self.canvas.canvasy(event.y)

    def _on_render_text_click(self):
        if self.current_img_index == -1:
            return

        active_bundle = self.gallery_images[self.current_img_index]
        original_img_rgb = active_bundle.get("img_rgb")
        img_inpainted = active_bundle.get("img_inpainted")
        text_regions = active_bundle.get("text_regions", [])
        text_strings = active_bundle.get("text_strings", [])

        if img_inpainted is None or original_img_rgb is None:
            return

        for i, region in enumerate(text_regions):
            if i < len(text_strings):
                region.translation = text_strings[i]

        if self.active_layer_index is not None and self.active_layer_index < len(text_regions):
            text_regions[self.active_layer_index].fg_color = self.current_fg_color
            text_regions[self.active_layer_index].font_color = self.current_fg_color
            text_regions[self.active_layer_index].bg_color = self.current_bg_color
            text_regions[self.active_layer_index].stroke_color = self.current_bg_color

        try:
            #set_font(os.path.join(BASE_PATH, 'fonts/msyh.ttc'))

            new_rendered_layers, new_coords = render_textblock_list_eng_rendered(
                img=img_inpainted,
                text_regions=[text_regions[self.active_layer_index],], #text_regions,
                rendered_text_list=[],
                rendered_text_coords=[],
                original_img=original_img_rgb,
            )

            processed_layers = [
                layer if isinstance(layer, Image.Image) else Image.fromarray(layer)
                for layer in new_rendered_layers
            ]

            active_bundle["layers"][self.active_layer_index] = processed_layers[0]
            active_bundle["coords"][self.active_layer_index] = (new_coords[0][0], new_coords[0][1])
            #active_bundle["layers"] = processed_layers
            #active_bundle["coords"] = [(c[0], c[1]) for c in new_coords]

            if self.autosize_enabled.get():
                self.fit_background_to_window()
            else:
                self.load_background_image_by_index(self.current_img_index)
        except Exception as e:
            print(f"Error during on-demand text layer rendering: {e}")

    def _safe_append_thumbnail(self, data_bundle, thumb_tk):
        self.thumbnail_refs.append(thumb_tk)
        self.gallery_images.append(data_bundle)
        assigned_index = len(self.gallery_images) - 1

        btn = tk.Button(
            self.scrollable_gallery_frame,
            image=thumb_tk,
            command=lambda idx=assigned_index: self.load_background_image_by_index(idx),
            bd=2,
            relief="groove"
        )
        btn.pack(pady=8, padx=10)

        if len(self.gallery_images) == 1:
            self.load_background_image_by_index(0)

    def load_background_image_by_index(self, target_index: int):
        if not self.gallery_images or not (0 <= target_index < len(self.gallery_images)):
            return
        self.close_property_box()
        self.current_img_index = target_index
        self.load_background_image(self.gallery_images[target_index])

    def load_background_image(self, data_bundle: dict):
        pil_image = data_bundle["bg"]
        try:
            self.current_bg_pil = pil_image.copy()
            self.bg_width, self.bg_height = self.current_bg_pil.size
            target_width = self.bg_width + 150
            target_height = self.bg_height + 50

            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()

            if platform.system() == "Linux":
                try:
                    output = subprocess.check_output(
                        "xrandr | grep -E ' connected (primary )?[0-9]+x[0-9]+'",
                        shell=True,
                        stderr=subprocess.DEVNULL
                    ).decode("utf-8")
                    for part in output.split():
                        if "x" in part and "+" in part:
                            res_str = part.split("+")
                            dims = res_str[0].split("x")
                            screen_width = int(dims[0])
                            screen_height = int(dims[1])
                            break
                except Exception:
                    pass

            top_level = self.winfo_toplevel()

            if self.autosize_enabled.get():
                self.canvas_container.pack_propagate(True)
                final_width = min(target_width, screen_width) if screen_width > 0 else 1280
                final_height = min(target_height, screen_height) if screen_height > 0 else 720
                top_level.geometry(f"{final_width}x{final_height}")
                top_level.update_idletasks()
                self.canvas_container.pack_propagate(False)
                self.fit_background_to_window()
            else:
                self.canvas_container.pack_propagate(True)
                self.tk_bg_image = ImageTk.PhotoImage(self.current_bg_pil)
                final_width = min(target_width, int(screen_width * 0.75)) if screen_width > 0 else 1280
                final_height = min(target_height, int(screen_height * 0.75)) if screen_height > 0 else 720
                top_level.geometry(f"{final_width}x{final_height}")

                self.canvas.delete("all")
                self.layer_image_refs.clear()
                self.active_layer_id = None
                self.canvas.config(scrollregion=(0, 0, self.bg_width, self.bg_height))
                self.canvas.create_image(0, 0, image=self.tk_bg_image, anchor="nw", tags="bg_layer")
                self.load_draggable_text_image_layers(data_bundle)

        except Exception as e:
            print(f"Error executing background transition: {e}")

    def load_draggable_text_image_layers(self, data_bundle: dict):
        self._tk_overlay_cache.clear()
        layers = data_bundle.get("layers", [])
        coords = data_bundle.get("coords", [])

        for i, pil_layer in enumerate(layers):
            if not isinstance(pil_layer, Image.Image):
                continue
            try:
                tk_layer_img = ImageTk.PhotoImage(pil_layer)
                self._tk_overlay_cache.append(tk_layer_img)

                x, y = coords[i] if i < len(coords) else (50, 50)
                layer_tag = f"layer_idx_{i}"
                layer_id = self.canvas.create_image(
                    x, y, image=tk_layer_img, anchor="nw", tags=("draggable_layer", layer_tag)
                )

                self.layer_image_refs[layer_id] = tk_layer_img
                self.active_layer_id = layer_id

                self.canvas.tag_bind(layer_id, "<ButtonPress-1>", lambda event, lid=layer_id, lidx=i: self._on_drag_start(event, lid, lidx))
                self.canvas.tag_bind(layer_id, "<B1-Motion>", self._on_dragging)
                self.canvas.tag_bind(layer_id, "<ButtonRelease-1>", lambda event, lid=layer_id, lidx=i: self._on_drag_end(event, lid, lidx))
            except Exception as e:
                print(f"Error rendering layer {i}: {e}")

    def _on_drag_end(self, event, target_id, layer_index):
        if self.autosize_enabled.get() or self.current_img_index == -1:
            return

        current_coords = self.canvas.coords(target_id)
        if current_coords:
            active_bundle = self.gallery_images[self.current_img_index]
            coords = active_bundle.setdefault("coords", [])
            while len(coords) <= layer_index:
                coords.append((50, 50))
            coords[layer_index] = (int(current_coords[0]), int(current_coords[1]))

    def _on_autosize_toggle_click(self):
        is_autosize = self.autosize_enabled.get()

        btn_state = "disabled" if is_autosize else "normal"
        self.export_img_btn.config(state=btn_state)
        self.export_all_btn.config(state=btn_state)

        if is_autosize:
            self.canvas_container.pack_propagate(False)
            self.fit_background_to_window()
        else:
            self.canvas_container.pack_propagate(True)
            if self.current_img_index != -1:
                self.load_background_image_by_index(self.current_img_index)

    def _on_window_resize_event(self, event):
        if self.autosize_enabled.get():
            if abs(event.width - (self.bg_width + 4)) > 5 or abs(event.height - (self.bg_height + 4)) > 5:
                self.fit_background_to_window()

    def _on_arrow_left_press(self, event):
        if not self.gallery_images or self.current_img_index == -1:
            return
        new_index = (self.current_img_index - 1) % len(self.gallery_images)
        self.load_background_image_by_index(new_index)

    def _on_arrow_right_press(self, event):
        if not self.gallery_images or self.current_img_index == -1:
            return
        new_index = (self.current_img_index + 1) % len(self.gallery_images)
        self.load_background_image_by_index(new_index)

    def _on_sidebar_mousewheel(self, event):
        delta = -1 if (getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0) else 1
        self.gallery_canvas.yview_scroll(delta, "units")

    def _on_main_canvas_mousewheel(self, event):
        delta = -1 if (getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0) else 1
        if (event.state & 0x1) != 0:
            self.canvas.xview_scroll(delta, "units")
        else:
            self.canvas.yview_scroll(delta, "units")

    def register_close_protocol(self):
        self.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.on_close_cleanup)

    def on_close_cleanup(self):
        try:
            shutdown_event.set()
            self.close_property_box()
            self.canvas.delete("all")
            self.gallery_canvas.delete("all")
            self.layer_image_refs.clear()
            self._tk_overlay_cache.clear()
            self.thumbnail_refs.clear()
            self.tk_bg_image = None
            self.current_bg_pil = None
            self.winfo_toplevel().destroy()
        except Exception as e:
            print(f"Error during widget resource cleanup: {e}")

    def _on_dragging(self, event):
        if self.autosize_enabled.get() or not self.active_layer_id:
            return

        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.move(self.active_layer_id, cur_x - self.drag_data["x"], cur_y - self.drag_data["y"])
        self.drag_data["x"] = cur_x
        self.drag_data["y"] = cur_y

    def fit_background_to_window(self):
        if self.current_img_index == -1 or not self.current_bg_pil:
            return

        data_bundle = self.gallery_images[self.current_img_index]
        window_w = self.canvas_container.winfo_width()
        window_h = self.canvas_container.winfo_height()
        if window_w <= 1 or window_h <= 1:
            return

        view_w, view_h = max(window_w - 4, 50), max(window_h - 4, 50)

        try:
            self.canvas.config(width=view_w, height=view_h, bg="black")
            self.canvas.config(scrollregion=(0, 0, view_w, view_h))
            composite_canvas_img = Image.new("RGB", (view_w, view_h), color="black")

            img_w, img_h = self.current_bg_pil.size
            img_aspect, view_aspect = img_w / img_h, view_w / view_h

            '''if abs(img_aspect - view_aspect) > 0.3:
                new_w, new_h = view_w, view_h
                paste_x, paste_y = 0, 0
            else:'''
            if img_aspect > view_aspect:
                new_w = view_w
                new_h = int(view_w / img_aspect)
            else:
                new_h = view_h
                new_w = int(view_h * img_aspect)
            new_w, new_h = max(1, new_w), max(1, new_h)
            paste_x = (view_w - new_w) // 2
            paste_y = (view_h - new_h) // 2

            scaled_bg = self.current_bg_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            composite_canvas_img.paste(scaled_bg, (paste_x, paste_y))

            self.tk_bg_image = ImageTk.PhotoImage(composite_canvas_img)
            self.canvas.delete("bg_layer")
            self.canvas.create_image(0, 0, image=self.tk_bg_image, anchor="nw", tags="bg_layer")
            self.canvas.tag_lower("bg_layer")

            scale_x = new_w / img_w
            scale_y = new_h / img_h
            self.bg_width = view_w
            self.bg_height = view_h

            self._tk_overlay_cache.clear()
            self.canvas.delete("draggable_layer")

            layers = data_bundle.get("layers", [])
            coords = data_bundle.get("coords", [])

            for i, pil_layer in enumerate(layers):
                if not isinstance(pil_layer, Image.Image):
                    continue
                try:
                    layer_w, layer_h = pil_layer.size
                    new_l_w = max(1, int(layer_w * scale_x))
                    new_l_h = max(1, int(layer_h * scale_y))

                    scaled_layer_pil = pil_layer.resize((new_l_w, new_l_h), Image.Resampling.LANCZOS)
                    tk_layer_img = ImageTk.PhotoImage(scaled_layer_pil)
                    self._tk_overlay_cache.append(tk_layer_img)

                    orig_x, orig_y = coords[i] if i < len(coords) else (50, 50)
                    translated_x = int(orig_x * scale_x) + paste_x
                    translated_y = int(orig_y * scale_y) + paste_y

                    layer_id = self.canvas.create_image(
                        translated_x, translated_y, image=tk_layer_img, anchor="nw", tags="draggable_layer"
                    )
                    self.layer_image_refs[layer_id] = tk_layer_img
                    self.active_layer_id = layer_id

                    self.canvas.tag_bind(layer_id, "<ButtonPress-1>", lambda event, lid=layer_id, lidx=i: self._on_drag_start(event, lid, lidx))
                    self.canvas.tag_bind(layer_id, "<B1-Motion>", self._on_dragging)
                    self.canvas.tag_bind(layer_id, "<ButtonRelease-1>", lambda event, lid=layer_id, lidx=i: self._on_drag_end(event, lid, lidx))
                except Exception as e:
                    print(f"Error scaling layer {i}: {e}")

        except Exception as e:
            print(f"Autosize resizing exception: {e}")


def enlarge_window(rect, im_w, im_h, ratio=2.5, aspect_ratio=1.0) -> List:
    assert ratio > 1.0
    
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]

    # https://numpy.org/doc/stable/reference/generated/numpy.roots.html
    coeff = [aspect_ratio, w+h*aspect_ratio, (1-ratio)*w*h]
    roots = np.roots(coeff)
    roots.sort()
    delta = int(round(roots[-1] / 2))
    delta_w = int(delta * aspect_ratio)
    delta_w = min(x1, im_w - x2, delta_w)
    delta = min(y1, im_h - y2, delta)
    rect = np.array([x1-delta_w, y1-delta, x2+delta_w, y2+delta], dtype=np.int64)
    rect[::2] = np.clip(rect[::2], 0, im_w - 1)
    rect[1::2] = np.clip(rect[1::2], 0, im_h - 1)
    return rect.tolist()

def extract_ballon_region(img: np.ndarray, ballon_rect: List, enlarge_ratio=1, verbose=False) -> Tuple[np.ndarray, int, List]:

    x1, y1, x2, y2 = ballon_rect[0], ballon_rect[1], ballon_rect[2] + ballon_rect[0], ballon_rect[3] + ballon_rect[1]
    if enlarge_ratio > 1:
        x1, y1, x2, y2 = enlarge_window([x1, y1, x2, y2], img.shape[1], img.shape[0], enlarge_ratio, aspect_ratio=ballon_rect[3] / ballon_rect[2])

    img = img[y1:y2, x1:x2].copy()

    kernel = np.ones((3,3), np.uint8)
    orih, oriw = img.shape[0], img.shape[1]
    scaleR = 1
    if orih > 300 and oriw > 300:
        scaleR = 0.6
    elif orih < 120 or oriw < 120:
        scaleR = 1.4

    if scaleR != 1:
        h, w = img.shape[0], img.shape[1]
        orimg = np.copy(img)
        img = cv2.resize(img, (int(w*scaleR), int(h*scaleR)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[0], img.shape[1]
    img_area = h * w

    cpimg = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
    detected_edges = cv2.Canny(cpimg, 70, 140, L2gradient=True, apertureSize=3)
    cv2.rectangle(detected_edges, (0, 0), (w-1, h-1), WHITE, 1, cv2.LINE_8)
    cons, hiers = cv2.findContours(detected_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.rectangle(detected_edges, (0, 0), (w-1, h-1), BLACK, 1, cv2.LINE_8)

    ballon_mask = np.zeros((h, w), np.uint8)
    min_retval = np.inf
    mask = np.zeros((h, w), np.uint8)
    difres = 10
    seedpnt = (int(w/2), int(h/2))
    for i in range(len(cons)):
        rect = cv2.boundingRect(cons[i])
        if rect[2]*rect[3] < img_area*0.4:
            continue

        mask = cv2.drawContours(mask, cons, i, (255), 2)
        cpmask = np.copy(mask)
        cv2.rectangle(mask, (0, 0), (w-1, h-1), WHITE, 1, cv2.LINE_8)
        retval, _, _, rect = cv2.floodFill(cpmask, mask=None, seedPoint=seedpnt, flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))

        if retval <= img_area * 0.3:
            mask = cv2.drawContours(mask, cons, i, (0), 2)
        if retval < min_retval and retval > img_area * 0.3:
            min_retval = retval
            ballon_mask = cpmask

    ballon_mask = 127 - ballon_mask
    ballon_mask = cv2.dilate(ballon_mask, kernel,iterations = 1)
    ballon_area, _, _, rect = cv2.floodFill(ballon_mask, mask=None, seedPoint=seedpnt, flags=4, newVal=(30), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
    ballon_mask = 30 - ballon_mask    
    retval, ballon_mask = cv2.threshold(ballon_mask, 1, 255, cv2.THRESH_BINARY)
    ballon_mask = cv2.bitwise_not(ballon_mask, ballon_mask)

    box_kernel = int(np.sqrt(ballon_area) / 30)
    if box_kernel > 1:
        box_kernel = np.ones((box_kernel,box_kernel),np.uint8)
        ballon_mask = cv2.dilate(ballon_mask, box_kernel, iterations = 1)
        ballon_mask = cv2.erode(ballon_mask, box_kernel, iterations = 1)

    if scaleR != 1:
        img = orimg
        ballon_mask = cv2.resize(ballon_mask, (oriw, orih))

    if verbose:
        cv2.imshow('ballon_mask', ballon_mask)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    return ballon_mask, [x1, y1, x2, y2]
