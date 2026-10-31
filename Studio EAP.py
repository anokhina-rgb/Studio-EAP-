import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import zipfile
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.stats import gaussian_kde

# Створення головного вікна додатку
root = tk.Tk()
root.title("Visualiser Studio for English for Academic Purposes (EAP)")
root.geometry("760x520")
root.config(bg="#f8fafc")

# Професійний заголовок інтерфейсу
title_label = tk.Label(
    root,
    text="📊 Visualiser Studio for English for Academic Purposes (EAP)",
    font=("Arial", 12, "bold"),
    bg="#f8fafc",
    fg="#1e293b",
)
title_label.pack(pady=15)

# Фрейм вибору модулів
frame_select = tk.Frame(root, bg="#f8fafc")
frame_select.pack(pady=5)

tk.Label(
    frame_select,
    text="Select Analytical Module (Lessons 1–13):",
    font=("Arial", 11, "bold"),
    bg="#f8fafc",
    fg="#334155",
).pack(anchor="w", padx=10, pady=2)

lesson_types = [
    "1. Bar Charts (Comparative Frequencies)",
    "2. Line Plots (Diachronic Longitudinal Trends)",
    "3. Scatter Plots (Bivariate Token Length vs. TTR)",
    "4. Dot Plots (Discrete Modal Frequencies)",
    "5. Box Plots (Cross-Validation Dispersion)",
    "6. Dendrograms (Hierarchical Text Clusters)",
    "7. Histograms (Sentence Length Distribution)",
    "8. CART (Classification & Regression Trees)",
    "9. Pie Charts (Translation Strategy Proportions)",
    "10. Strip Plots (Sentence Length Point Dispersion)",
    "11. Mosaic Charts (Categorical Associations)",
    "12. Density Plots (Kernel Density Estimation)",
    "13. Area Charts (Diachronic Compositional Shares)",
]

combo = ttk.Combobox(
    frame_select, values=lesson_types, width=64, font=("Arial", 10)
)
combo.set(lesson_types[0])
combo.pack(padx=10, pady=8)


# Допоміжна функція для створення безпечної та унікальної назви файлу
def generate_safe_filename(selected_text, extension):
  base = "".join(
      c if c.isalnum() or c in (" ", "_", "-") else "" for c in selected_text
  )
  base = "_".join(base.split()).lower()
  return f"{base}.{extension}"


# Повна функція побудови графіків для всіх 13 типів без скорочень
def draw_chart(ax, selected, alpha_mod=1.0):
  if "1. Bar Charts" in selected:
    cats = [
        "Nominalization",
        "Passive Voice",
        "Hedging Markers",
        "Attribution Verbs",
    ]
    s1 = [88.0 * alpha_mod, 72.0, 65.0, 54.0]
    s2 = [75.0, 60.0 * alpha_mod, 58.0, 48.0]
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w / 2, s1, w, label="DIY Student Corpus", color="#2563eb")
    ax.bar(
        x + w / 2, s2, w, label="Standard Academic (Brown)", color="#64748b"
    )
    ax.set_ylabel("Normalized Frequency", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=10)
    ax.legend(frameon=True)

  elif "2. Line Plots" in selected:
    cats = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
    s1 = [65.0, 68.0, 72.0, 76.0, 82.0, 85.0, 88.0]
    s2 = [80.0, 78.0, 75.0, 71.0, 68.0, 63.0, 60.0]
    ax.plot(
        cats,
        [v * alpha_mod for v in s1],
        marker="o",
        label="DIY Student Corpus",
        color="#2563eb",
        linewidth=2.5,
    )
    ax.plot(
        cats,
        s2,
        marker="s",
        label="Standard Academic",
        color="#dc2626",
        linewidth=2.5,
    )
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.legend(frameon=True)

  elif "3. Scatter Plots" in selected:
    x_v = [400, 850, 1200, 1900, 2400, 3100, 4200, 4900]
    y_v = [
        v * alpha_mod
        for v in [0.42, 0.55, 0.64, 0.71, 0.76, 0.81, 0.84, 0.86]
    ]
    ax.scatter(
        x_v,
        y_v,
        color="#2563eb",
        s=90,
        edgecolor="black",
        label="Analyzed Texts",
    )
    ax.set_xlabel("Token Count", fontweight="bold")
    ax.set_ylabel("TTR", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

  elif "4. Dot Plots" in selected:
    modals = ["Must", "Should", "Could", "Would", "May", "Might"]
    vals = [int(45 * alpha_mod), 38, 28, 22, 15, 8]
    ax.scatter(modals, vals, color="#2563eb", s=140, zorder=3, label="Frequency")
    for m, v in zip(modals, vals):
      ax.vlines(
          m, 0, v, color="#94a3b8", linestyle="--", linewidth=1.5, zorder=2
      )
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_ylim(0, 55)
    ax.grid(axis="y", linestyle=":", alpha=0.7)

  elif "5. Box Plots" in selected:
    data = [
        [12 * alpha_mod, 15, 18, 20, 25, 28],
        [8, 10, 14, 16, 22, 24],
        [20, 22, 25, 30, 35, 38],
    ]
    ax.boxplot(
        data,
        tick_labels=["Sub-corpus A", "Sub-corpus B", "Sub-corpus C"],
        patch_artist=True,
        boxprops=dict(facecolor="#93c5fd", color="#2563eb"),
    )
    ax.set_ylabel("Sentence Length", fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

  elif "6. Dendrograms" in selected:
    np.random.seed(42)
    X = np.random.rand(8, 2) * alpha_mod
    Z = sch.linkage(X, method="ward")
    sch.dendrogram(
        Z, ax=ax, labels=[f"T{i+1}" for i in range(8)], color_threshold=0.7
    )
    ax.set_ylabel("Euclidean Distance", fontweight="bold")

  elif "7. Histograms" in selected:
    np.random.seed(42)
    lengths = np.random.normal(loc=22 * alpha_mod, scale=5, size=250)
    ax.hist(
        lengths, bins=12, color="#3b82f6", edgecolor="#1e40af", alpha=0.85
    )
    ax.set_xlabel("Words per Sentence", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

  elif "8. CART" in selected:
    ax.axis("off")
    ax.text(
        0.5,
        0.8,
        f"Root Split: POS > {0.45 * alpha_mod:.2f}\n(Gini = 0.48)",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.6", fc="#93c5fd", ec="#2563eb"),
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.25,
        0.35,
        "Academic EAP",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.6", fc="#cbd5e1", ec="#475569"),
        fontsize=9,
    )
    ax.text(
        0.75,
        0.35,
        "Student DIY",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.6", fc="#cbd5e1", ec="#475569"),
        fontsize=9,
    )
    ax.plot([0.5, 0.25], [0.7, 0.52], "k--", lw=1.5)
    ax.plot([0.5, 0.75], [0.7, 0.52], "k--", lw=1.5)

  elif "9. Pie Charts" in selected:
    cats = [
        "Literal Translation",
        "Transposition",
        "Modulation",
        "Adaptation",
        "Omission",
    ]
    props = [42.0 * alpha_mod, 28.0, 15.0, 10.0, 5.0]
    colors = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#cbd5e1"]
    ax.pie(
        props,
        labels=cats,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(edgecolor="white"),
    )
    ax.axis("equal")

  elif "10. Strip Plots" in selected:
    corpora = ["Corpus A", "Corpus B", "Corpus C"]
    np.random.seed(42)
    for i, name in enumerate(corpora):
      y_vals = np.random.normal(loc=(20 + i * 4) * alpha_mod, scale=3.5, size=25)
      x_vals = np.random.normal(loc=i + 1, scale=0.06, size=25)
      ax.scatter(x_vals, y_vals, color="#2563eb", alpha=0.7, s=45)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(corpora)
    ax.set_ylabel("Sentence Length", fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

  elif "11. Mosaic Charts" in selected:
    rect1 = plt.Rectangle(
        (0, 0), 0.6 * alpha_mod, 0.7 * alpha_mod, fc="#2563eb", ec="white"
    )
    rect2 = plt.Rectangle((0.6, 0), 0.4, 0.7, fc="#60a5fa", ec="white")
    rect3 = plt.Rectangle((0, 0.7), 0.3, 0.3, fc="#93c5fd", ec="white")
    rect4 = plt.Rectangle((0.3, 0.7), 0.7, 0.3, fc="#cbd5e1", ec="white")
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    ax.text(
        0.3,
        0.35,
        "Noun / Formal",
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

  elif "12. Density Plots" in selected:
    np.random.seed(10)
    d1 = np.random.normal(20 * alpha_mod, 3, 120)
    d2 = np.random.normal(26, 3.5, 120)
    grid = np.linspace(8, 38, 200)
    kde1 = gaussian_kde(d1)
    kde2 = gaussian_kde(d2)
    ax.plot(grid, kde1(grid), color="#2563eb", lw=2.5, label="Student")
    ax.fill_between(grid, kde1(grid), color="#2563eb", alpha=0.3)
    ax.plot(grid, kde2(grid), color="#dc2626", lw=2.5, label="Academic")
    ax.fill_between(grid, kde2(grid), color="#dc2626", alpha=0.3)
    ax.legend(frameon=True)

  elif "13. Area Charts" in selected:
    time_p = ["1990", "1995", "2000", "2005", "2010", "2015", "2020"]
    y1 = [v * alpha_mod for v in [10, 15, 25, 40, 55, 70, 85]]
    y2 = [5, 10, 18, 30, 45, 60, 75]
    ax.fill_between(
        time_p, y1, color="#3b82f6", alpha=0.6, label="Literal Share"
    )
    ax.fill_between(
        time_p, y2, color="#93c5fd", alpha=0.6, label="Transposition Share"
    )
    ax.set_ylabel("Share (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True)

  if "CART" not in selected and "Mosaic" not in selected:
    ax.set_title(selected, fontweight="bold", pad=12)
  elif "CART" in selected:
    ax.set_title(selected, fontweight="bold", pad=12)


# Експорт поточного елемента у PNG
def export_png():
  selected = combo.get()
  folder_selected = filedialog.askdirectory(title="Select Destination Folder")
  if not folder_selected:
    return

  fig, ax = plt.subplots(figsize=(8, 5))
  draw_chart(ax, selected, alpha_mod=1.0)
  fig.tight_layout()

  file_name = generate_safe_filename(selected, "png")
  filename = os.path.join(folder_selected, file_name)

  fig.savefig(filename, dpi=300)
  plt.close(fig)

  messagebox.showinfo(
      "PNG Export Successful",
      f"High-resolution image automatically saved as:\n'{file_name}'",
  )


# Експорт поточного елемента у GIF
def export_gif():
  selected = combo.get()
  folder_selected = filedialog.askdirectory(
      title="Select Destination Folder for GIF"
  )
  if not folder_selected:
    return

  fig, ax = plt.subplots(figsize=(8, 5))
  file_name = generate_safe_filename(selected, "gif")
  filename = os.path.join(folder_selected, file_name)

  def update(frame):
    ax.clear()
    alpha = 0.5 + (frame / 10.0)
    draw_chart(ax, selected, alpha_mod=alpha)
    ax.set_title(
        f"{selected}\n[Frame {frame+1}/10]", fontweight="bold", pad=12
    )

  ani = animation.FuncAnimation(fig, update, frames=10, interval=150, repeat=True)
  try:
    ani.save(filename, writer="pillow", fps=6)
    plt.close(fig)
    messagebox.showinfo(
        "GIF Export Successful",
        f"Animated loop automatically saved as:\n'{file_name}'",
    )
  except Exception as e:
    plt.close(fig)
    messagebox.showerror("Error", f"Could not save GIF: {str(e)}")


# Експорт УСІХ 13 типів (PNG та GIF) в один ZIP-архів
def export_all_zip():
  folder_selected = filedialog.askdirectory(
      title="Select Destination Folder for ZIP Archive"
  )
  if not folder_selected:
    return

  zip_filename = os.path.join(folder_selected, "eap_visualiser_studio_all.zip")

  try:
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
      with tempfile.TemporaryDirectory() as tmpdir:
        for item in lesson_types:
          # 1. Генерація та додавання PNG
          fig, ax = plt.subplots(figsize=(8, 5))
          draw_chart(ax, item, alpha_mod=1.0)
          fig.tight_layout()
          png_name = generate_safe_filename(item, "png")
          png_path = os.path.join(tmpdir, png_name)
          fig.savefig(png_path, dpi=300)
          plt.close(fig)
          zf.write(png_path, arcname=f"png_high_res/{png_name}")

          # 2. Генерація та додавання GIF
          fig, ax = plt.subplots(figsize=(8, 5))
          gif_name = generate_safe_filename(item, "gif")
          gif_path = os.path.join(tmpdir, gif_name)

          def update_maker(curr_item):
            def update(frame):
              ax.clear()
              alpha = 0.5 + (frame / 10.0)
              draw_chart(ax, curr_item, alpha_mod=alpha)
              ax.set_title(
                  f"{curr_item}\n[Frame {frame+1}/10]",
                  fontweight="bold",
                  pad=12,
              )

            return update

          ani = animation.FuncAnimation(
              fig,
              update_maker(item),
              frames=10,
              interval=150,
              repeat=True,
          )
          ani.save(gif_path, writer="pillow", fps=6)
          plt.close(fig)
          zf.write(gif_path, arcname=f"animated_gifs/{gif_name}")

    messagebox.showinfo(
        "ZIP Archive Successful",
        (
            "All 13 analytical modules (PNG & GIF) successfully bundled into"
            f" archive:\n'{os.path.basename(zip_filename)}'"
        ),
    )
  except Exception as e:
    messagebox.showerror("Error", f"Could not create ZIP archive: {str(e)}")


# Панель кнопок управління
frame_btn = tk.Frame(root, bg="#f8fafc")
frame_btn.pack(pady=20)

btn_png = tk.Button(
    frame_btn,
    text="📷 Export PNG",
    command=export_png,
    bg="#2563eb",
    fg="white",
    font=("Arial", 9, "bold"),
    padx=10,
    pady=8,
)
btn_png.pack(side="left", padx=6)

btn_gif = tk.Button(
    frame_btn,
    text="🎞️ Export GIF",
    command=export_gif,
    bg="#059669",
    fg="white",
    font=("Arial", 9, "bold"),
    padx=10,
    pady=8,
)
btn_gif.pack(side="left", padx=6)

btn_zip = tk.Button(
    frame_btn,
    text="📦 Export All to ZIP",
    command=export_all_zip,
    bg="#7c3aed",
    fg="white",
    font=("Arial", 9, "bold"),
    padx=10,
    pady=8,
)
btn_zip.pack(side="left", padx=6)

# Професійна підказка внизу
footer = tk.Label(
    root,
    text=(
        "Visualiser Studio for English for Academic Purposes (EAP) — Automated"
        " Export Utility"
    ),
    font=("Arial", 9, "italic"),
    bg="#f8fafc",
    fg="#64748b",
)
footer.pack(pady=5)

root.mainloop()