# make_summary_plot.py
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

ax.text(0.5, 0.95, "Σύνοψη Πολυπλοκότητας (H1) — EMG → Κινηματικά → GRF",
        ha='center', va='center', fontsize=16, weight='bold')

cols = ["EMG", "Κινηματικά (Angles)", "Δυναμικά (GRF)"]
x_positions = [0.15, 0.50, 0.85]
for x, c in zip(x_positions, cols):
    ax.text(x, 0.88, c, ha='center', va='center', fontsize=12)

rows = ["Healthy", "Post-stroke"]
y_positions = [0.65, 0.35]
for y, r in zip(y_positions, rows):
    ax.text(0.02, y, r, ha='left', va='center', fontsize=12, style='italic')

healthy_texts = [
    "H1: μέτρια\n(σταθερός έλεγχος)",
    "H1: υψηλή\n(πλούσια μορφή)",
    "H1: υψηλή\n(διπλή κορυφή, καθαρό μοτίβο)"
]
post_texts = [
    "H1: υψηλή\n(ασταθές, υπερ-πολύπλοκο EMG)",
    "H1: χαμηλή\n(«επίπεδη» άρθρωση)",
    "H1: χαμηλή\n(επίπεδη φόρτιση)"
]

box_w, box_h = 0.22, 0.14

def draw_row(y_center, texts):
    for x, t in zip(x_positions, texts):
        rect = plt.Rectangle((x - box_w/2, y_center - box_h/2), box_w, box_h, fill=False)
        ax.add_patch(rect)
        ax.text(x, y_center, t, ha='center', va='center', fontsize=11)
    for i in range(len(x_positions)-1):
        x0 = x_positions[i] + box_w/2
        x1 = x_positions[i+1] - box_w/2
        ax.annotate("", xy=(x1, y_center), xytext=(x0, y_center),
                    arrowprops=dict(arrowstyle="->", lw=1.2))

draw_row(y_positions[0], healthy_texts)
draw_row(y_positions[1], post_texts)

ax.text(0.5, 0.12, "Ερμηνεία H1: χαμηλή → απλή μορφή | υψηλή → πολύπλοκη/πλούσια μορφή", ha='center', fontsize=11)
ax.text(0.5, 0.06, "Στους post-stroke η πολυπλοκότητα μένει στο EMG και δεν μεταφράζεται σε κίνηση/GRF.",
        ha='center', fontsize=11)

plt.savefig("complexity_flow_summary.png", dpi=200, bbox_inches='tight')
print("Saved: complexity_flow_summary.png")