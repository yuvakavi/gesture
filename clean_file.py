with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Keep only the first 496 lines
clean_lines = lines[:496]

with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(clean_lines)
