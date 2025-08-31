


# config.py
DB_HOST = "localhost"
DB_NAME = "hospital_system"
DB_USER = "postgres"
DB_PASSWORD = "shahinda"

# Themes
DARK_THEME = """
QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
}
QPushButton {
    background-color: #3c3f41;
    border: none;
    padding: 5px;
}
QPushButton:hover {
    background-color: #505354;
}
QTableView {
    background-color: #3c3f41;
    gridline-color: #5c5c5c;
}
"""

LIGHT_THEME = """
QWidget {
    background-color: #f0f0f0;
    color: #000000;
}
QPushButton {
    background-color: #e0e0e0;
    border: none;
    padding: 5px;
}
QPushButton:hover {
    background-color: #d0d0d0;
}
QTableView {
    background-color: #ffffff;
    gridline-color: #cccccc;
}
"""