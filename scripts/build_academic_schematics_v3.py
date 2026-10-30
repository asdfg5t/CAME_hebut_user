"""Twelve standalone, compact academic diagrams. No training data or slide layouts."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'PPT汇总/配图/学术示意图_重绘版'
QA = ROOT / 'tmp/weekly_report_20261016/academic_schematics_v3'
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
FONT = 'C:/Windows/Fonts/msyh.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'mathtext.fontset': 'stix',
                     'svg.fonttype': 'path', 'axes.unicode_minus': False})
INK = '#202932'
BLUE, ORANGE, PURPLE, GREEN = '#205D82', '#994B24', '#5B4778', '#28604C'
GRAY = '#626C76'
PALE_BLUE, PALE_ORANGE, PALE_PURPLE = '#E8F0F5', '#F7ECE5', '#EFEAF5'
LINE = '#65717B'
RECORDS = []
AX = None
LABELS = []
W = H = None


def canvas(width, height):
    global AX, LABELS, W, H
    W, H = width, height
    f, AX = plt.subplots(figsize=(width/100, height/100), dpi=100)
    f.subplots_adjust(left=0, right=1, top=1, bottom=0)
    AX.set(xlim=(0, width), ylim=(height, 0), aspect='equal')
    AX.axis('off')
    LABELS = []
    return f


def txt(x, y, value, size=25, color=INK, weight='normal', align='center', box=False):
    a = AX.text(x, y, value, fontsize=size*.72, color=color, ha=align, va='center',
                fontweight=weight, zorder=5,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.1} if box else None)
    LABELS.append((a, size))
    return a


def line(p, q, color=LINE, width=1.65, dashed=False):
    AX.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=width,
            ls=(0, (4, 3)) if dashed else '-', zorder=1)


def arrow(p, q, color=INK, dashed=False, width=1.85):
    AX.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=12,
                  color=color, linewidth=width, shrinkA=0, shrinkB=0,
                  linestyle=(0, (4, 3)) if dashed else '-', zorder=2))


def rect(cx, cy, width, height, value=None, color=INK, fill='white', size=25, dashed=False):
    AX.add_patch(Rectangle((cx-width/2, cy-height/2), width, height,
                  fc=fill, ec=color, lw=1.65, linestyle=(0, (4, 3)) if dashed else '-', zorder=3))
    if value is not None:
        txt(cx, cy, value, size, color)


def circ(x, y, radius, value=None, color=INK, fill='white', size=27):
    AX.add_patch(Circle((x, y), radius, fc=fill, ec=color, lw=1.65, zorder=3))
    if value is not None:
        txt(x, y, value, size, color)


def strip(x, y, values, color=BLUE, cell=21, height=32, numeric=False, mask=()):
    """Symbolic feature cells, never presented as measured data."""
    for j, v in enumerate(values):
        left = x+j*cell
        masked = j in mask
        AX.add_patch(Rectangle((left, y), cell-1, height, fc='#D8DDE1' if masked else 'white',
                              ec=INK if masked else color, lw=1.0, zorder=3))
        if masked:
            AX.plot([left+4,left+cell-5],[y+5,y+height-5],color=INK,lw=1.3,zorder=4)
            AX.plot([left+4,left+cell-5],[y+height-5,y+5],color=INK,lw=1.3,zorder=4)
        else:
            AX.add_patch(Rectangle((left+.7, y+height*(1-v)), cell-2.4, height*v-.7,
                                  fc=color, ec='none', alpha=.22, zorder=3))
        if numeric:
            txt(left+cell/2, y+height/2, str(v), 23)


def graph(cx, cy, active=None, labels=False, color=BLUE, scale=1):
    coords = [(0, 39), (-32, 2), (32, 2), (-47, -40), (32, -40), (52, -79)]
    edges = [(0,1),(0,2),(1,3),(2,4),(4,5)]
    dist = [0,1,1,2,2,3]
    for u, v in edges:
        p = (cx+coords[u][0]*scale, cy+coords[u][1]*scale)
        q = (cx+coords[v][0]*scale, cy+coords[v][1]*scale)
        line(p, q, LINE, 1.5)
    for i, (dx, dy) in enumerate(coords):
        on = active is not None and dist[i] <= active
        c = color if on or active is None else GRAY
        circ(cx+dx*scale, cy+dy*scale, (14 if i==0 else 11)*scale,
             chr(65+i) if labels or i == 0 else None, c,
             PALE_BLUE if on else 'white', 21)


def tiny_graph(cx, cy, color=INK, dropped_edge=False):
    p = [(cx-24,cy+11),(cx,cy-15),(cx+25,cy+12)]
    for u, v in [(0,1),(1,2),(0,2)]:
        if dropped_edge and (u,v)==(0,2):
            continue
        line(p[u], p[v], color, 1.55)
    for x, y in p:
        circ(x,y,6,color=color)


def save(f, name, page, concept, evidence, example=None, notes=None, target=420):
    """Check all text extents before exporting; raster QA uses actual display sizes."""
    f.canvas.draw()
    renderer = f.canvas.get_renderer()
    bounds_errors = []
    for a, size in LABELS:
        b = a.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W+1 or b.y1 > H+1:
            bounds_errors.append(a.get_text())
    if bounds_errors:
        raise ValueError(f'{name}: text outside canvas: {bounds_errors}')
    files = {}
    for ext in ['svg','png']:
        path = OUT / f'{name}.{ext}'
        f.savefig(path, dpi=400, transparent=True, pad_inches=0)
        files[ext] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    plt.close(f)
    img = Image.open(OUT / f'{name}.png').convert('RGBA')
    white = Image.new('RGBA', img.size, 'white')
    white.alpha_composite(img)
    for width in [320, target]:
        thumb = white.convert('RGB').resize((width, round(width*H/W)), Image.Resampling.LANCZOS)
        thumb.save(QA / f'{name}_white_{width}px.png')
    RECORDS.append({'name':name,'page':page,'concept':concept,'basis':evidence,
                    'synthetic_example':example,'notes':notes,'view_box':[W,H],
                    'review_width_px':target,'minimum_label_px_at_review_width':round(min(s for _,s in LABELS)*target/W,1),
                    'files':files})


def graph_matrices():
    f = canvas(510,240)
    p = [(32,69),(119,69),(32,182),(119,182)]
    for u,v in [(0,1),(0,2),(1,3)]:
        line(p[u],p[v],BLUE if (u,v)==(0,1) else INK,2.3 if (u,v)==(0,1) else 1.65)
    for i,(x,y) in enumerate(p):
        circ(x,y,15,str(i+1),BLUE if i<2 else INK,size=25)
    txt(75,23,r'$\mathcal{G}$',31)
    arrow((143,122),(169,122))
    txt(248,23,r'$A$',31)
    matrix = [[0,1,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,0]]
    for i in range(4):
        for j in range(4):
            hi=(i,j) in [(0,1),(1,0)]
            rect(195+j*33,72+i*33,32,32,str(matrix[i][j]),BLUE if hi else INK,
                 PALE_BLUE if hi else 'white',25)
    txt(420,23,r'$X$',31)
    for i in range(4):
        txt(350,72+i*33,r'$x_'+str(i+1)+'$',26)
        for j in range(4):
            AX.add_patch(Rectangle((374+j*30,56+i*33),28,32,
                         fc=GREEN,alpha=.17+.09*((i+j)%4),ec='white',lw=.8,zorder=3))
    txt(247,216,r'$A_{12}=A_{21}=1$',25,BLUE)
    save(f,'01_graph_matrix_features',2,'图与矩阵的对应关系','Attributed graph notation',
         {'adjacency':matrix,'edge_pairs':[[1,2],[1,3],[2,4]]},
         'Feature shading is symbolic, not measured values.')


def messages():
    f=canvas(435,225)
    for y,s,c in [(38,r'$h_B=4$',BLUE),(111,r'$h_A=2$',PURPLE),(184,r'$h_C=6$',BLUE)]:
        txt(67,y,s,31,c)
    arrow((120,41),(171,91),BLUE)
    arrow((120,111),(166,111),PURPLE)
    arrow((120,181),(171,131),BLUE)
    rect(217,111,90,72,color=GREEN,fill='#EEF3F0')
    txt(217,95,'均值',25,GREEN)
    txt(217,124,r'$\mu$',33,GREEN)
    arrow((270,111),(305,111))
    txt(369,111,r"$h_A'=4$",32,PURPLE)
    txt(277,194,r'$(2+4+6)/3=4$',28)
    save(f,'02_message_passing',5,'邻居和中心节点的均值聚合','Gilmer et al., ICML 2017; generic teaching example',
         {'self':2,'neighbors':[4,6],'mean':4},'Identity messages and direct mean update; not GCN degree normalization.')


def receptive_field():
    f=canvas(470,247)
    for k,cx in enumerate([73,235,397]):
        txt(cx,22,f'{k}层',27)
        graph(cx,132,active=k,scale=.92)
    arrow((137,119),(164,119),GRAY)
    arrow((299,119),(326,119),GRAY)
    txt(235,224,'高亮：可影响 A 的节点',24)
    save(f,'03_gcn_receptive_field',6,'局部消息传递的传播范围','Kipf & Welling, ICLR 2017; local message passing',
         {'target':'A','hops':[0,1,2]},'Highlights are possible information sources, not attention scores or performance.')


def contrastive_views():
    f=canvas(510,258)
    for y,s,c,fill in [(65,'a',BLUE,PALE_BLUE),(195,'b',ORANGE,PALE_ORANGE)]:
        tiny_graph(42,y,c,dropped_edge=(s=='b'))
        txt(43,y-43,r'$\mathcal{G}^{('+s+')}$',29,c)
        arrow((79,y),(110,y),c)
        rect(169,y,105,57,r'$f_\theta$',c,fill,32)
        arrow((230,y),(266,y),c)
        strip(274,y-18,[.7,.35,.85] if s=='a' else [.65,.4,.8],c,23,36)
        txt(308,y-45,r'$z_i^{('+s+')}$',29,c)
        arrow((351,y),(407,116 if s=='a' else 145),c)
    txt(168,130,'共享参数',23)
    rect(453,130,96,70,color=PURPLE,fill=PALE_PURPLE)
    txt(453,114,'对比',25,PURPLE)
    txt(453,143,r'$\mathcal{L}_{\rm con}$',28,PURPLE)
    save(f,'04_self_supervised_learning',13,'两个相关图视图的对比学习','Generic GraphCL / GRACE background',
         notes='The lower view removes one edge as a generic augmentation example. Encoders share parameters. Feature cells are symbolic, not experimental measurements. Not the full CAME architecture.')


def masked_reconstruction():
    f=canvas(500,173)
    strip(10,62,[.4,.7,.3,.6],BLUE,23,40,mask=(1,3))
    txt(56,26,r'$\widetilde{x}_i$',32)
    txt(56,133,'遮蔽输入',24)
    arrow((111,82),(136,82))
    rect(193,82,100,60,r'$f_\theta$',BLUE,PALE_BLUE,32)
    txt(193,133,'编码器',24)
    arrow((251,82),(276,82))
    rect(333,82,100,60,r'$g_\phi$',GREEN,'#EEF3F0',32)
    txt(333,133,'解码器',24)
    arrow((391,82),(416,82))
    strip(425,62,[.5,.8,.3,.6],GREEN,17,40)
    txt(458,26,r'$\widehat{x}_i$',32)
    txt(452,133,'重建属性',23)
    save(f,'04b_masked_reconstruction',12,'遮蔽属性后重建','Generic reconstruction self-supervision / GraphMAE background',
         notes='Crossed cells indicate masked coordinates; values are symbolic, not measurements.')


def matching():
    f=canvas(350,313)
    for i in range(3):
        txt(47,76+i*61,r'$a_'+str(i+1)+'$',29,BLUE if i==0 else INK)
        txt(112+i*61,25,r'$b_'+str(i+1)+'$',29,ORANGE)
        for j in range(3):
            rect(112+j*61,76+i*61,59,59,r'$s_{'+str(i+1)+str(j+1)+'}$',
                 BLUE if i==j else LINE,PALE_BLUE if i==j else 'white',28)
    AX.add_patch(Rectangle((79,43),187,65,fc='none',ec=BLUE,lw=2.7,zorder=4))
    rect(31,280,17,17,color=BLUE,fill=PALE_BLUE)
    txt(108,280,'同一节点',23)
    rect(204,280,17,17,color=LINE)
    txt(280,280,'其他节点',23)
    save(f,'05_contrastive_matching',14,'正配对与候选负配对','InfoNCE background',
         notes='Diagonal is same-node matching; scores are symbolic; highlighted row is anchor a1.',target=350)


def encoding():
    f=canvas(550,255)
    for y,m,c,fill,label in [(57,'t',BLUE,PALE_BLUE,'Text'),(198,'v',ORANGE,PALE_ORANGE,'Image')]:
        txt(29,y,'文本' if m=='t' else '图像',24,c)
        arrow((57,y),(83,y),c)
        rect(134,y,92,65,color=c,fill=fill)
        txt(134,y-14,'CLIP',24,c)
        txt(134,y+15,label,24,c)
        arrow((188,y),(212,y),c)
        txt(247,y,r'$X^{('+m+')}$',31,c)
        arrow((279,y),(304,y),c)
        rect(365,y,106,65,r'$\mathrm{GNN}_{'+m+'}$',c,fill,29)
        arrow((426,y),(454,y),c)
        txt(506,y,r'$Z^{('+m+')}$',33,c)
    tiny_graph(364,128,GREEN)
    arrow((364,106),(364,96),GREEN)
    arrow((364,149),(364,158),GREEN)
    txt(430,129,r'$\mathcal{G}$',30,GREEN)
    save(f,'06_clip_graph_alignment',18,'同拓扑下的两路模态专属编码','CAME Eq. (3)-(5)',
         notes='CLIP is applied node-wise to collect matrix X. GNN parameters are modality-specific; topology is shared. Outputs Z collect node embeddings.',target=480)


def tri_view():
    f=canvas(360,280)
    # Undirected dashed segments mean loss constraints, never data flow.
    line((94,195),(160,108),PURPLE,1.75,True)
    line((266,195),(200,108),PURPLE,1.75,True)
    line((107,218),(253,218),PURPLE,1.75,True)
    txt(180,39,r'$z_i^{(f)}$',37,PURPLE)
    txt(180,83,'融合',24,PURPLE)
    txt(65,211,r'$z_i^{(t)}$',37,BLUE)
    txt(65,254,'文本',24,BLUE)
    txt(295,211,r'$z_i^{(v)}$',37,ORANGE)
    txt(295,254,'视觉',24,ORANGE)
    txt(91,134,r'$\mathcal{L}_{t,f}$',29,PURPLE,box=True)
    txt(269,134,r'$\mathcal{L}_{v,f}$',29,PURPLE,box=True)
    txt(180,218,r'$\mathcal{L}_{t,v}$',29,PURPLE,box=True)
    save(f,'06b_tri_view_alignment',17,'同一节点的三对视图约束','CAME Eq. (16)-(18)',
         notes='Dashed lines represent pairwise objectives, not data flow or a claim of symmetric loss.',target=360)


def fusion_order():
    f=canvas(520,300)
    txt(19,75,'a',25,GRAY)
    for y,m,c in [(35,'t',BLUE),(113,'v',ORANGE)]:
        txt(86,y,r'$X^{('+m+')}$',30,c)
        arrow((122,y),(164,62 if m=='t' else 88),c)
    circ(193,75,28,r'$\mu$',PURPLE,PALE_PURPLE,34)
    arrow((230,75),(269,75))
    rect(330,75,105,58,'GNN',size=27)
    arrow((393,75),(447,75))
    txt(482,75,r'$Z$',33)
    txt(19,218,'b',25,GRAY)
    for y,m,c,fill in [(176,'t',BLUE,PALE_BLUE),(262,'v',ORANGE,PALE_ORANGE)]:
        txt(86,y,r'$X^{('+m+')}$',30,c)
        arrow((122,y),(159,y),c)
        rect(217,y,105,57,r'$\mathrm{GNN}_{'+m+'}$',c,fill,28)
        arrow((279,y),(335,205 if m=='t' else 231),c)
    rect(386,218,90,60,'融合',PURPLE,PALE_PURPLE,26)
    arrow((440,218),(458,218))
    txt(489,218,r'$Z$',33)
    save(f,'07_fusion_order',19,'先平均后传播与独立传播后融合','CAME motivation; average as one early-fusion example',
         notes='a: mean then GNN. b: modality-specific GNNs then fusion. No performance conclusion.',target=480)


def noninjective():
    f=canvas(505,230)
    for y,left,right in [(49,'[1,0]','[0,1]'),(179,'[0.5,0.5]','[0.5,0.5]')]:
        txt(62,y,left,26,BLUE)
        txt(177,y,right,26,ORANGE)
        arrow((243,y),(283,95 if y<100 else 135),PURPLE)
    circ(312,115,28,r'$\mu$',PURPLE,PALE_PURPLE,34)
    arrow((349,115),(374,115))
    txt(438,115,'[0.5,0.5]',28)
    txt(312,171,'平均',24,PURPLE)
    save(f,'07b_average_noninjective',20,'不同输入组合具有相同平均值','Synthetic algebraic counterexample',
         {'inputs':[[[1,0],[0,1]],[[.5,.5],[.5,.5]]],'same_mean':[.5,.5]},
         'Shows noninjectivity, not a universal downstream performance claim.',target=420)


def gating():
    f=canvas(550,270)
    txt(64,24,r'$z^{(t)}$',31,BLUE)
    txt(64,90,'[1,0]',31,BLUE)
    txt(64,180,'[0,1]',31,ORANGE)
    txt(64,247,r'$z^{(v)}$',31,ORANGE)
    for y,c in [(90,BLUE),(180,ORANGE)]:
        arrow((112,y),(167,y),c)
        circ(197,y,24,r'$\odot$',c,size=32)
        arrow((228,y),(326,122 if y<100 else 150),c)
    txt(245,23,r'$g=[0.8,0.2]$',26,PURPLE)
    arrow((197,41),(197,58),PURPLE)
    txt(260,248,r'$1-g=[0.2,0.8]$',26,PURPLE)
    arrow((197,229),(197,213),PURPLE)
    circ(358,136,25,'+',PURPLE,PALE_PURPLE,35)
    arrow((392,136),(420,136))
    txt(480,88,r'$z^{(g)}$',32,PURPLE)
    txt(480,136,'[0.8,0.8]',29,PURPLE)
    save(f,'08_gating_and_experts',22,'逐维门控的两路加权与求和','CAME Eq. (6)-(7)',
         {'text':[1,0],'image':[0,1],'gate':[.8,.2],'output':[.8,.8]},
         'The gate is a vector. This is the intermediate gated embedding, not final expert fusion.',target=480)


def expert_routing():
    f=canvas(535,285)
    txt(24,142,r'$z$',34)
    arrow((46,142),(72,142))
    rect(133,142,104,64,'路由器',size=25)
    txt(133,201,'Top-2',25)
    for y,i,on in [(43,1,True),(142,2,False),(241,3,True)]:
        rect(310,y,92,56,r'$\phi_'+str(i)+'$',GREEN if on else GRAY,
             '#EEF3F0' if on else 'white',33,dashed=not on)
        if on:
            arrow((194,142),(254,y),GREEN)
            arrow((367,y),(428,126 if i==1 else 158),GREEN)
            txt(407,73 if i==1 else 211,'0.7' if i==1 else '0.3',25,GREEN,box=True)
    circ(454,142,24,r'$\Sigma$',GREEN,size=32)
    arrow((487,142),(505,142),GREEN)
    txt(523,142,r'$e$',30,GREEN)
    save(f,'08b_expert_routing',23,'单个专家组内的Top-k路由','CAME intra-group expert routing; teaching example',
         {'expert_count':3,'top_k':2,'weights':[.7,.3]},
         'Counts and weights are illustrative, not CAME hyperparameters. Unselected expert has no active path.',target=480)


def package():
    manifest={'revision':'academic-schematics-v3','date':'2026-10-16',
              'scope':'12 standalone educational figures; no slides or experiment protocol changes',
              'background':'transparent SVG and PNG; all QA on white',
              'text':'outlined in SVG to avoid font substitution',
              'figures':RECORDS}
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    note='独立学术示意图 重绘版\n\nPNG为高分辨率图，SVG为矢量图。字形已转为路径，避免字体替换。\n所有数值均为教学示例，论文依据和含义边界见figure_manifest.json。\n仅重绘配图，Word、MD和论文原图均未修改。\n\n'
    for r in RECORDS:
        note+=f'第{r["page"]:02d}页 {r["concept"]}：{r["name"]}.svg\n'
    (OUT/'配图对应页码.txt').write_text(note,encoding='utf-8-sig')
    # Preview is a contact sheet only; individual assets contain no headings or captions.
    font=ImageFont.truetype(FONT,19)
    sheet=Image.new('RGB',(1120,4*350+20),'#EDF0F2')
    draw=ImageDraw.Draw(sheet)
    for n,r in enumerate(RECORDS):
        col,row=n%3,n//3
        x,y=10+col*370,10+row*350
        draw.rectangle((x,y,x+359,y+339),fill='white')
        draw.text((x+11,y+10),f'{r["page"]:02d}  {r["concept"]}',font=font,fill=INK)
        p=QA/f'{r["name"]}_white_320px.png'
        im=Image.open(p)
        # Keep 320 px actual width; no upscaling hides small-label defects.
        sheet.paste(im,(x+(360-im.width)//2,y+48+(285-im.height)//2))
    sheet.save(OUT/'配图总览_白底实际尺寸.png')
    archive=OUT/'独立学术示意图_重绘版_SVG与PNG.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for r in RECORDS:
            for ext in ['svg','png']:
                z.write(OUT/f'{r["name"]}.{ext}',f'第{r["page"]:02d}页_{r["concept"]}.{ext}')
        for fn in ['figure_manifest.json','配图对应页码.txt']:
            z.write(OUT/fn,fn)
    print(f'Generated {len(RECORDS)} SVG/PNG pairs. Review at white 320 px and per-figure widths.')


if __name__=='__main__':
    graph_matrices();messages();receptive_field();contrastive_views();masked_reconstruction()
    matching();encoding();tri_view();fusion_order();noninjective();gating();expert_routing()
    package()
