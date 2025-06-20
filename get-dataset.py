from roboflow import Roboflow

# Ganti dengan API KEY milik kamu
rf = Roboflow(api_key="SxTU5QzMcMDZuCXO1ixo")
project = rf.workspace("platnomorpa").project("platnomorpa")
dataset = project.version(1).download("yolov8")