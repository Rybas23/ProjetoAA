import time
import tkinter as tk
from tkinter import scrolledtext


class Visualizador:
    # Símbolos para visualização
    SYMBOLS = {
        "empty": ".",
        "wall": "W",
        "resource": "R",
        "ninho": "N",
        "farol": "F",
    }

    def __init__(self, grid_width, grid_height, title="Ambiente", fps=5, cell_size=40):
        self.width = grid_width
        self.height = grid_height
        self.fps = fps
        self.running = True
        self.title = title

        # Componentes Tkinter (lazy init)
        self._root = None
        self._text = None

    # ==================== TKINTER SETUP ====================

    def _init_tk(self):
        """Inicializa a janela Tkinter apenas uma vez."""
        if self._root is not None:
            return

        self._root = tk.Tk()
        self._root.title(self.title)

        # Caixa de texto tipo consola
        self._text = scrolledtext.ScrolledText(
            self._root,
            width=self.width + 4,   # margem para bordas
            height=self.height + 4,  # margem
            font=("Consolas", 12),
        )
        self._text.pack(fill=tk.BOTH, expand=True)

        # Impedir edição pelo utilizador
        self._text.config(state=tk.DISABLED)

        # Quando o utilizador fecha a janela, parámos o rendering
        def on_close():
            self.running = False
            self._root.destroy()
            self._root = None

        self._root.protocol("WM_DELETE_WINDOW", on_close)

    # ==================== LÓGICA DE GRELHA ====================

    def _create_empty_grid(self):
        return [
            [self.SYMBOLS["empty"] for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def _is_valid_position(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def _place_on_grid(self, grid, x, y, symbol):
        if self._is_valid_position(x, y):
            grid[y][x] = symbol

    def _draw_elements(self, grid, elements, symbol):
        if not elements:
            return
        for pos in elements:
            if isinstance(pos, tuple) and len(pos) == 2:
                x, y = pos
                self._place_on_grid(grid, x, y, symbol)

    def _draw_single_element(self, grid, position, symbol):
        if position and isinstance(position, tuple) and len(position) == 2:
            x, y = position
            self._place_on_grid(grid, x, y, symbol)

    def _draw_agents(self, grid, agents):
        for agent_id, (x, y) in agents.items():
            if self._is_valid_position(x, y):
                symbol = str(agent_id)[0].upper()
                grid[y][x] = symbol

    # ==================== RENDER NA JANELA ====================

    def _print_grid(self, grid, agents):
        """Atualiza a janela Tkinter com o conteúdo ASCII."""

        # Garante que a janela existe
        self._init_tk()
        if self._root is None or self._text is None:
            return

        # Montar o texto em memória (string única)
        lines = []
        lines.append(f"=== {self.title} ===")
        lines.append("+" + "-" * self.width + "+")
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * self.width + "+")

        full_text = "\n".join(lines)

        # Atualizar widget de texto
        self._text.config(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.insert(tk.END, full_text)
        self._text.config(state=tk.DISABLED)

        # Atualizar a janela (não bloqueante)
        self._root.update_idletasks()
        self._root.update()

        # Desacelerar de acordo com fps (frames por segundo)
        if self.fps > 0:
            time.sleep(1.0 / self.fps)

    # ==================== API PÚBLICA ====================

    def draw_grid(self, resources, agents, ninho=None, farol=None, walls=None):
        if not self.running:
            return False

        grid = self._create_empty_grid()

        # Ordem de desenho
        self._draw_elements(grid, walls, self.SYMBOLS["wall"])
        self._draw_elements(grid, resources.keys(), self.SYMBOLS["resource"])
        self._draw_single_element(grid, ninho, self.SYMBOLS["ninho"])
        self._draw_single_element(grid, farol, self.SYMBOLS["farol"])
        self._draw_agents(grid, agents)

        self._print_grid(grid, agents)
        return True

    def draw(self, ambiente):
        recursos = getattr(ambiente, "resources", {})
        agentes = ambiente.agent_pos
        ninho = getattr(ambiente, "ninho", None)
        farol = getattr(ambiente, "farol", None)
        walls = getattr(ambiente, "walls", None)  # funciona para Farol e Foraging

        return self.draw_grid(recursos, agentes, ninho=ninho, farol=farol, walls=walls)

    def check_events(self):
        return self.running

    def cleanup(self):
        if self._root is not None:
            self._root.destroy()
            self._root = None
        self.running = False
