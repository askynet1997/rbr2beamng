from __future__ import annotations

import tkinter as tk
import unicodedata
from tkinter import ttk

import pypdfium2 as pdfium
from PIL import ImageTk

PdfError = pdfium.PdfiumError
_PAGE_GAP = 10
_RENDER_MARGIN_PAGES = 1


def pdf_text(data: bytes) -> str:
    document = pdfium.PdfDocument(data)
    try:
        pages = []
        for index in range(len(document)):
            page = document[index]
            text = page.get_textpage()
            pages.append(text.get_text_range())
            text.close()
            page.close()
        return "\n".join(pages)
    finally:
        document.close()


def _fold(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    ).casefold()


def find_text(
    document: pdfium.PdfDocument,
    query: str,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Case- and accent-insensitive matches as (page index, (left, bottom, right, top)) in PDF points."""
    needle = _fold(query.strip())
    matches: list[tuple[int, tuple[float, float, float, float]]] = []
    if not needle:
        return matches
    for index in range(len(document)):
        page = document[index]
        text = page.get_textpage()
        folded: list[str] = []
        owners: list[int] = []
        for position, char in enumerate(text.get_text_range(0, text.count_chars())):
            for folded_char in _fold(char):
                folded.append(folded_char)
                owners.append(position)
        haystack = "".join(folded)
        start = haystack.find(needle)
        while start >= 0:
            boxes = [
                box
                for box in (
                    text.get_charbox(char)
                    for char in range(owners[start], owners[start + len(needle) - 1] + 1)
                )
                if box[2] > box[0] or box[3] > box[1]
            ]
            if boxes:
                matches.append(
                    (
                        index,
                        (
                            min(box[0] for box in boxes),
                            min(box[1] for box in boxes),
                            max(box[2] for box in boxes),
                            max(box[3] for box in boxes),
                        ),
                    )
                )
            start = haystack.find(needle, start + len(needle))
        text.close()
        page.close()
    return matches


class PdfViewer(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        data: bytes,
        query: str,
        *,
        background: str,
        page_border: str,
        match_color: str,
        current_color: str,
    ) -> None:
        super().__init__(master)
        self.document = pdfium.PdfDocument(data)
        self.page_sizes = [self.document.get_page_size(index) for index in range(len(self.document))]
        self.match_color = match_color
        self.current_color = current_color
        self.page_border = page_border
        self.scale = 0.0
        self.page_tops: list[float] = []
        self.page_left = 0.0
        self.rendered: dict[int, tuple[int, ImageTk.PhotoImage]] = {}
        self.matches: list[tuple[int, tuple[float, float, float, float]]] = []
        self.current = -1
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=(0, 6))
        ttk.Label(toolbar, text="Find").grid(row=0, column=0, padx=(0, 6))
        self.query = tk.StringVar(value=query)
        entry = ttk.Entry(toolbar, textvariable=self.query, width=36)
        entry.grid(row=0, column=1)
        entry.bind("<Return>", lambda _event: self.search())
        ttk.Button(toolbar, text="Search", command=self.search).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(toolbar, text="◀", width=3, command=lambda: self.show_match(-1)).grid(row=0, column=3, padx=(6, 0))
        ttk.Button(toolbar, text="▶", width=3, command=lambda: self.show_match(1)).grid(row=0, column=4, padx=(4, 0))
        self.status = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.status).grid(row=0, column=5, padx=(8, 0))

        self.canvas = tk.Canvas(
            self,
            background=background,
            highlightthickness=0,
            yscrollincrement=40,
        )
        self.canvas.grid(row=1, column=0, sticky=tk.NSEW)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._yview)
        scrollbar.grid(row=1, column=1, sticky=tk.NS)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Destroy>", self._on_destroy)
        self._find_matches()

    def search(self) -> None:
        self._find_matches()
        self._draw_matches()
        self._scroll_to_current()

    def show_match(self, step: int) -> None:
        if not self.matches:
            return
        self.current = (self.current + step) % len(self.matches)
        self._update_status()
        self._draw_matches()
        self._scroll_to_current()

    def _find_matches(self) -> None:
        self.matches = find_text(self.document, self.query.get())
        self.current = 0 if self.matches else -1
        self._update_status()

    def _update_status(self) -> None:
        if not self.query.get().strip():
            self.status.set("")
        elif self.matches:
            self.status.set(f"{self.current + 1} of {len(self.matches)} matches")
        else:
            self.status.set("No matches")

    def _on_resize(self, event: tk.Event) -> None:
        width = max(event.width - 2 * _PAGE_GAP, 50)
        scale = width / max(page_width for page_width, _height in self.page_sizes)
        if abs(scale - self.scale) < 0.01:
            self._render_visible()
            return
        first_layout = not self.scale
        self.scale = scale
        self._layout()
        if first_layout:
            self._scroll_to_current()
        self._render_visible()

    def _layout(self) -> None:
        self.canvas.delete("all")
        self.rendered.clear()
        self.page_tops = []
        top = float(_PAGE_GAP)
        self.page_left = float(_PAGE_GAP)
        for page_width, page_height in self.page_sizes:
            self.page_tops.append(top)
            self.canvas.create_rectangle(
                self.page_left,
                top,
                self.page_left + page_width * self.scale,
                top + page_height * self.scale,
                outline=self.page_border,
                fill="white",
            )
            top += page_height * self.scale + _PAGE_GAP
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), top))
        self._draw_matches()

    def _match_rectangle(self, index: int) -> tuple[float, float, float, float]:
        page, (left, bottom, right, top) = self.matches[index]
        page_top = self.page_tops[page]
        page_height = self.page_sizes[page][1]
        return (
            self.page_left + left * self.scale - 2,
            page_top + (page_height - top) * self.scale - 2,
            self.page_left + right * self.scale + 2,
            page_top + (page_height - bottom) * self.scale + 2,
        )

    def _draw_matches(self) -> None:
        self.canvas.delete("match")
        if not self.page_tops:
            return
        for index in range(len(self.matches)):
            current = index == self.current
            self.canvas.create_rectangle(
                *self._match_rectangle(index),
                outline=self.current_color if current else self.match_color,
                width=3 if current else 2,
                tags=("match",),
            )

    def _scroll_to_current(self) -> None:
        if self.current < 0 or not self.page_tops:
            return
        _left, top, _right, _bottom = self._match_rectangle(self.current)
        total = float(self.canvas.cget("scrollregion").split()[3])
        self.canvas.yview_moveto(max(0.0, top - self.canvas.winfo_height() / 3) / total)
        self._render_visible()

    def _yview(self, *args) -> None:
        self.canvas.yview(*args)
        self._render_visible()

    def _on_mouse_wheel(self, event: tk.Event) -> str:
        self.canvas.yview_scroll(-3 * int(event.delta / 120), "units")
        self._render_visible()
        return "break"

    def _render_visible(self) -> None:
        if not self.page_tops:
            return
        top = self.canvas.canvasy(0)
        bottom = top + self.canvas.winfo_height()
        visible = [
            index
            for index, page_top in enumerate(self.page_tops)
            if page_top < bottom and page_top + self.page_sizes[index][1] * self.scale > top
        ]
        if not visible:
            return
        wanted = range(
            max(visible[0] - _RENDER_MARGIN_PAGES, 0),
            min(visible[-1] + _RENDER_MARGIN_PAGES + 1, len(self.page_sizes)),
        )
        for index in list(self.rendered):
            if index not in wanted:
                self.canvas.delete(self.rendered.pop(index)[0])
        for index in wanted:
            if index in self.rendered:
                continue
            page = self.document[index]
            image = page.render(scale=self.scale).to_pil()
            page.close()
            photo = ImageTk.PhotoImage(image, master=self.canvas)
            item = self.canvas.create_image(
                self.page_left,
                self.page_tops[index],
                anchor=tk.NW,
                image=photo,
            )
            self.canvas.tag_raise("match")
            self.rendered[index] = (item, photo)

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            self.rendered.clear()
            self.document.close()
