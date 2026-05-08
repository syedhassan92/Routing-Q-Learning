"""
main.py
========
Entry point for the Network Routing Simulator.
Launches the interactive GUI where users can build topologies,
run Dijkstra and Q-Learning, and simulate congestion visually.

Usage:  python main.py
"""

from gui_app import App


def main():
    """Launch the GUI application."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
