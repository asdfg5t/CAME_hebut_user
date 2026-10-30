"""A compact message-passing example readable at a 320 px display width."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image, ImageOps

root = Path(__file__).resolve().parents[1]
out = root / 'PPT汇总/配图/白底清晰SVG'
qa = root / 'tmp/weekly_report_20261016/message_passing_readability'
qa.mkdir(parents=True, exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'svg.fonttype': 'path'})
ink, blue, purple, green = '#17212B', '#195476', '#51416D', '#245A46'

# One canvas unit is one pixel in the readability design. No outer padding.
fig, ax = plt.subplots(figsize=(4.2, 2.2), dpi=100)
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 420), ylim=(220, 0), aspect='equal')
ax.axis('off')

def label(x, y, value, size, color=ink, weight='normal'):
    # Font sizes in points: convert from design pixels to physical points.
    ax.text(x, y, value, ha='center', va='center', fontsize=size * .72,
            color=color, fontweight=weight)

def node(x, y, name, value, color, fill):
    ax.add_patch(Circle((x, y), 32, facecolor=fill, edgecolor=color, linewidth=1.9))
    label(x, y-13, name, 23, color, 'bold')
    label(x, y+13, str(value), 27, ink, 'bold')

def arrow(start, end, color):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12,
                                linewidth=2.0, color=color, shrinkA=0, shrinkB=0))

node(42, 45, 'B', 4, blue, '#E5EFF5')
node(42, 175, 'C', 6, blue, '#E5EFF5')
node(151, 110, 'A', 2, purple, '#EDE9F3')
arrow((70, 61), (121, 92), blue)
arrow((70, 159), (121, 128), blue)
arrow((188, 110), (226, 110), ink)
ax.add_patch(Circle((262, 110), 30, facecolor='white', edgecolor=green, linewidth=1.9))
label(262, 109, 'μ', 38, green)
label(262, 54, '均值', 25, green)
arrow((298, 110), (337, 110), ink)
node(377, 110, 'A′', 4, purple, '#EDE9F3')
label(266, 185, '(2 + 4 + 6) ÷ 3 = 4', 23, ink)

stem = '02_message_passing_小尺寸清晰版'
for extension in ['svg', 'png']:
    fig.savefig(out / f'{stem}.{extension}', dpi=400, transparent=True, pad_inches=0)
plt.close(fig)

original = Image.open(out / f'{stem}.png').convert('RGBA')
white = Image.new('RGBA', original.size, 'white')
white.alpha_composite(original)
white.convert('RGB').resize((320, 168), Image.Resampling.LANCZOS).save(qa / 'white_background_at_320px.png')
manifest = {'figure': stem, 'numeric_example': {'B': 4, 'C': 6, 'A': 2, 'updated_A': 4},
            'aggregation': 'mean of self and two neighbors',
            'not_gcn_degree_normalization': True, 'not_experiment_data': True,
            'qa_display_size_px': [320, 168], 'minimum_label_size_at_320px': 17.5,
            'changes': 'compact geometry; large labels and values; thicker strokes; original color families retained',
            'files': {ext: hashlib.sha256((out/f'{stem}.{ext}').read_bytes()).hexdigest() for ext in ['svg','png']}}
(qa / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved compact SVG and high-resolution PNG; QA preview rendered at 320 x 168 px.')
