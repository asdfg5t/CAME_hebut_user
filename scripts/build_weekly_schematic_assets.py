"""Standalone vector schematics. No slide layouts or experiment data."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib import font_manager
from PIL import Image, ImageOps, ImageDraw, ImageFont
import sys
import re
from matplotlib.colors import to_rgba

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'PPT汇总' / '配图' / '学术示意图'
MONOCHROME = '--monochrome' in sys.argv
WHITE_BACKGROUND = '--white-background' in sys.argv
if MONOCHROME:
    OUT = ROOT / 'PPT汇总' / '配图' / '黑白配图' / '学术示意图'
elif WHITE_BACKGROUND:
    OUT = ROOT / 'PPT汇总' / '配图' / '白底清晰SVG'
FONT = 'C:/Windows/Fonts/msyh.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,
                     'mathtext.fontset':'dejavusans','svg.fonttype':'path'})
INK='#28313D'
MUTED='#9AA6B2'
EDGE='#BAC3CC'
BLUE='#367DA8'
ORANGE='#C4774C'
PURPLE='#8372A3'
GREEN='#4F8A78'
PALE_BLUE='#E5EFF5'
PALE_ORANGE='#F6EBE3'
PALE_PURPLE='#EDE9F3'
RECORDS=[]


def apply_white_background_style(f):
    """Retain semantic color fills; darken text and strokes for white slides."""
    colors = {
        INK: '#17212B', MUTED: '#505A65', EDGE: '#59636E',
        BLUE: '#195476', ORANGE: '#8A411C', PURPLE: '#51416D', GREEN: '#245A46',
    }
    mapping = {to_rgba(k)[:3]: v for k, v in colors.items()}
    for a in f.axes:
        for label in a.texts:
            old = to_rgba(label.get_color())
            label.set_color(mapping.get(old[:3], '#17212B'))
            label.set_fontsize(max(9.5, label.get_fontsize()))
        for stroke in a.lines:
            old = to_rgba(stroke.get_color())
            stroke.set_color(mapping.get(old[:3], '#59636E'))
            stroke.set_linewidth(max(1.5, stroke.get_linewidth()))
        for patch in a.patches:
            edge = patch.get_edgecolor()
            if edge[3] > 0:
                patch.set_edgecolor(mapping.get(edge[:3], edge))
                patch.set_linewidth(max(1.15, patch.get_linewidth()))
            if isinstance(patch, FancyArrowPatch):
                face = patch.get_facecolor()
                if face[3] > 0:
                    patch.set_facecolor(mapping.get(face[:3], face))
                patch.set_linewidth(max(1.5, patch.get_linewidth()))


def apply_monochrome_style(f):
    """Keep geometry and values; use black labels and patterned visual features."""
    for a in f.axes:
        for label in a.texts:
            label.set_color('black')
            label.set_fontsize(max(9.5, label.get_fontsize()))
            if re.fullmatch(r'\d+(\.\d+)?', label.get_text()):
                label.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'pad': .15})
        for stroke in a.lines:
            old = to_rgba(stroke.get_color())
            stroke.set_color('black')
            stroke.set_linewidth(max(1.65, stroke.get_linewidth()))
            if old == to_rgba(ORANGE):
                stroke.set_linestyle('--')
        for patch in a.patches:
            old_edge = patch.get_edgecolor()
            old_face = patch.get_facecolor()
            alpha = patch.get_alpha()
            patch.set_alpha(None)
            if old_edge[3] > 0:
                patch.set_edgecolor('black')
                patch.set_linewidth(max(1.3, patch.get_linewidth()))
                if old_edge[:3] == to_rgba(ORANGE)[:3] and isinstance(patch, FancyArrowPatch):
                    patch.set_linestyle('--')
            if isinstance(patch, FancyArrowPatch):
                patch.set_edgecolor('black')
                patch.set_facecolor('black')
                continue
            if old_face[3] == 0:
                patch.set_facecolor('none')
            elif old_face[:3] == (1, 1, 1):
                patch.set_facecolor('white')
            elif old_face[:3] in [to_rgba(ORANGE)[:3], to_rgba(PALE_ORANGE)[:3]]:
                patch.set_facecolor('white')
                patch.set_edgecolor('black')
                patch.set_hatch('///')
            else:
                if alpha is not None:
                    tone = max(.58, 1 - alpha * .75)
                elif old_face[:3] == to_rgba(PALE_PURPLE)[:3]:
                    tone = .65
                elif old_face[:3] == to_rgba(PALE_BLUE)[:3]:
                    tone = .83
                elif old_face[:3] == to_rgba('#F0F1F3')[:3]:
                    tone = .65
                elif old_face[:3] in [to_rgba('#F2F4F6')[:3], to_rgba('#F7F8FA')[:3]]:
                    tone = 1
                else:
                    tone = .9
                patch.set_facecolor(str(tone))
                if old_edge[3] == 0:
                    patch.set_edgecolor('none')
        a.set_facecolor('white')
    f.set_facecolor('white')

def figure(w,h):
    f,a=plt.subplots(figsize=(w,h))
    f.subplots_adjust(left=0,right=1,bottom=0,top=1)
    a.set(xlim=(0,w),ylim=(0,h),aspect='equal')
    a.axis('off')
    return f,a

def text(a,x,y,s,size=11,color=INK,**kw):
    return a.text(x,y,s,ha='center',va='center',fontsize=size,color=color,**kw)

def line(a,p,q,c=EDGE,lw=1.3,ls='-'):
    a.plot([p[0],q[0]],[p[1],q[1]],color=c,lw=lw,ls=ls,zorder=1)

def arrow(a,p,q,c=INK,lw=1.25,ls='-',rad=0,both=False):
    a.add_patch(FancyArrowPatch(p,q,arrowstyle='<->' if both else '-|>',
                mutation_scale=11,linewidth=lw,color=c,linestyle=ls,
                connectionstyle='arc3,rad='+str(rad),zorder=1))

def node(a,x,y,s,c=BLUE,r=.19,fill='white',size=11):
    a.add_patch(Circle((x,y),r,fc=fill,ec=c,lw=1.3,zorder=2))
    text(a,x,y,s,size,c,zorder=3)

def block(a,x,y,w,h,s,c=INK,fill='white',size=11,ls='-'):
    a.add_patch(Rectangle((x-w/2,y-h/2),w,h,fc=fill,ec=c,lw=1.15,ls=ls,zorder=2))
    text(a,x,y,s,size,c,zorder=3)

def vector(a,x,y,vals,c=BLUE,width=.24,height=.7,labels=True):
    n=len(vals)
    for i,v in enumerate(vals):
        a.add_patch(Rectangle((x+i*width,y),width,height,fc='white',ec=c,lw=.8,zorder=2))
        a.add_patch(Rectangle((x+i*width,y),width,max(.025,height*abs(v)),fc=c,alpha=.25,ec='none',zorder=2))
        if labels:text(a,x+(i+.5)*width,y+height/2,f'{v:g}',8,c,zorder=3)

def small_graph(a,cx,cy,scale=.65,active=0):
    coords=[(0,0),(-.8,.8),(.8,.8),(-1.3,1.7),(.8,1.7),(1.3,2.5)]
    edges=[(0,1),(0,2),(1,3),(2,4),(4,5)]
    dist=[0,1,1,2,2,3]
    for u,v in edges:
        p,q=coords[u],coords[v]
        line(a,(cx+p[0]*scale,cy+p[1]*scale),(cx+q[0]*scale,cy+q[1]*scale))
    for i,(x,y) in enumerate(coords):
        yes=dist[i]<=active
        node(a,cx+x*scale,cy+y*scale,chr(65+i),BLUE if yes else MUTED,
             r=.16,fill=PALE_BLUE if yes else 'white',size=10)

def save(f,name,concept,basis,example=None):
    if MONOCHROME:
        apply_monochrome_style(f)
    elif WHITE_BACKGROUND:
        apply_white_background_style(f)
    OUT.mkdir(parents=True,exist_ok=True)
    files=[]
    for ext in ['png','svg']:
        p=OUT/(name+'.'+ext)
        f.savefig(str(p),dpi=300,bbox_inches='tight',pad_inches=.12,transparent=not MONOCHROME)
        files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(f)
    RECORDS.append({'name':name,'concept':concept,'basis':basis,
                    'example':example,'style':'standalone schematic; no title, prose, or footer',
                    'files':files})

def graph_matrix():
    f,a=figure(8.2,2.8)
    p=[(.4,2.1),(1.65,2.1),(.4,.8),(1.65,.8)]
    for u,v in [(0,1),(0,2),(1,3)]:line(a,p[u],p[v],BLUE if (u,v)==(0,1) else EDGE,1.8)
    for i,(x,y) in enumerate(p):node(a,x,y,str(i+1),BLUE,r=.17)
    arrow(a,(2,1.45),(2.6,1.45))
    mat=[[0,1,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,0]]
    for i in range(4):
        for j in range(4):
            x,y=2.85+j*.41,2.13-i*.41
            hi=(i,j) in [(0,1),(1,0)]
            a.add_patch(Rectangle((x,y-.4),.4,.4,fc=PALE_BLUE if hi else '#F7F8FA',ec='white'))
            text(a,x+.2,y-.2,str(mat[i][j]),12,BLUE if hi else INK)
    text(a,3.65,2.65,r'$A$',15)
    text(a,1.0,2.65,r'$\mathcal{G}$',15)
    text(a,3.65,.16,r'$A_{12}=A_{21}=1$',11,BLUE)
    for i in range(4):
        text(a,5.3,1.94-i*.41,r'$x_'+str(i+1)+'$',12,GREEN)
        for j in range(5):
            a.add_patch(Rectangle((5.7+j*.32,1.76-i*.41),.3,.35,
                        fc=GREEN,alpha=.12+.12*((i+j)%4),ec='white'))
    text(a,6.05,2.65,r'$X$',15)
    line(a,(4.8,.6),(4.8,2.15),EDGE,lw=.8)
    save(f,'01_graph_matrix_features','图与矩阵对应','Attributed graph notation',{'edges':[[1,2],[1,3],[2,4]],'adjacency':mat,'feature_cells':'symbolic, not numeric data'})

def messages():
    f,a=figure(6.7,2.9)
    node(a,.5,2.3,'B',r=.23)
    node(a,.5,.5,'C',r=.23)
    node(a,1.85,1.4,'A',PURPLE,r=.25)
    arrow(a,(.73,2.21),(1.63,1.57),BLUE)
    arrow(a,(.73,.59),(1.63,1.23),BLUE)
    text(a,.5,2.75,r'$h_B=4$',11,BLUE)
    text(a,.5,.08,r'$h_C=6$',11,BLUE)
    text(a,1.85,.94,r'$h_A=2$',11,PURPLE)
    arrow(a,(2.15,1.4),(2.9,1.4))
    node(a,3.35,1.4,r'$\mu$',GREEN,r=.34,size=19)
    text(a,3.35,2.05,'Mean',10,GREEN)
    arrow(a,(3.75,1.4),(4.45,1.4))
    node(a,5.15,1.4,"A'",PURPLE,r=.25,size=12)
    text(a,5.15,2.05,r"$h_A'=4$",14,PURPLE)
    text(a,3.35,.45,r'$\frac{2+4+6}{3}$',16,GREEN)
    save(f,'02_message_passing','邻居消息聚合','Gilmer et al., ICML 2017',{'self':2,'neighbors':[4,6],'mean':4})

def receptive_field():
    f,a=figure(7.7,2.65)
    for k,cx in enumerate([1.25,3.85,6.45]):
        small_graph(a,cx,.5,.64,k)
        text(a,cx,2.48,[r'$H^{(0)}$',r'$H^{(1)}$',r'$H^{(2)}$'][k],13)
        text(a,cx,.08,['0-hop','1-hop','2-hop'][k],10,BLUE)
    arrow(a,(2.25,1.2),(2.65,1.2),MUTED)
    arrow(a,(4.85,1.2),(5.25,1.2),MUTED)
    save(f,'03_gcn_receptive_field','局部传播范围','Kipf & Welling, ICLR 2017',{'meaning':'highlighted nodes can influence A after the stated number of local layers; no performance claim'})

def ssl():
    f,a=figure(6.7,2.9)
    for y,c,s,z in [(2.1,BLUE,r'$\mathcal{G}^{(a)}$',r'$z_i^{(a)}$'),(.65,ORANGE,r'$\mathcal{G}^{(b)}$',r'$z_i^{(b)}$')]:
        for u,v in [((.25,y),(.8,y+.3)),((.8,y+.3),(1.05,y-.25))]:line(a,u,v,c)
        for x,yy in [(.25,y),(.8,y+.3),(1.05,y-.25)]:node(a,x,yy,'',c,r=.1)
        text(a,.65,y+.6,s,12,c)
        arrow(a,(1.2,y),(1.75,y),c)
        block(a,2.35,y,1.05,.55,r'$f_\theta$',c,PALE_BLUE if c==BLUE else PALE_ORANGE,size=14)
        arrow(a,(2.94,y),(3.55,y),c)
        vector(a,3.8,y-.25,[.7,.3,.8],c,labels=False,height=.5)
        text(a,4.15,y+.5,z,12,c)
        arrow(a,(4.6,y),(5.25,1.4),c)
    node(a,5.7,1.4,r'$\mathcal{L}_{\mathrm{con}}$',PURPLE,r=.39,size=13)
    save(f,'04_self_supervised_learning','两个相关视图的对比','GraphCL / GRACE, generic contrastive learning',{'encoder':'shared in this generic example; not CAME full architecture'})
    f,a=figure(6.4,1.9)
    for j in range(5):
        block(a,.25+j*.29,.95,.27,.52,'?' if j in [1,3] else '',MUTED,
              '#F0F1F3' if j in [1,3] else PALE_BLUE,size=11)
    text(a,.85,1.6,r'$\widetilde{x}_i$',13)
    arrow(a,(1.65,.95),(2.1,.95))
    block(a,2.65,.95,.9,.55,r'$f_\theta$',BLUE,PALE_BLUE,14)
    arrow(a,(3.2,.95),(3.65,.95))
    block(a,4.2,.95,.9,.55,r'$g_\phi$',GREEN,'#E8F0EC',14)
    arrow(a,(4.75,.95),(5.15,.95))
    vector(a,5.35,.69,[.5,.8,.3,.6,.4],GREEN,height=.52,labels=False,width=.15)
    text(a,5.6,1.6,r'$\widehat{x}_i$',13)
    save(f,'04b_masked_reconstruction','遮蔽与重建','Generic reconstruction SSL, GraphMAE background')

def contrastive():
    f,a=figure(4.5,3.9)
    x0,y0=.8,2.95
    for i in range(3):
        text(a,.35,y0-i*.75-.35,r'$a_'+str(i+1)+'$',12,BLUE)
        text(a,x0+i*.75+.35,3.45,r'$b_'+str(i+1)+'$',12,ORANGE)
        for j in range(3):
            x,y=x0+j*.75,y0-i*.75
            a.add_patch(Rectangle((x,y-.7),.7,.7,fc=PALE_PURPLE if i==j else '#F2F4F6',ec='white',lw=1.5))
            text(a,x+.35,y-.35,r'$s_{'+str(i+1)+str(j+1)+'}$',13,PURPLE if i==j else MUTED)
    a.add_patch(Rectangle((x0-.055,y0-.755),2.26,.81,fc='none',ec=BLUE,lw=1.3))
    arrow(a,(3.25,2.6),(3.65,2.6),BLUE)
    text(a,4.0,2.6,r'$a_1$',13,BLUE)
    a.add_patch(Rectangle((.85,.15),.18,.18,fc=PALE_PURPLE,ec=PURPLE,lw=.8))
    text(a,1.55,.24,'Positive',10,PURPLE)
    a.add_patch(Rectangle((2.4,.15),.18,.18,fc='#F2F4F6',ec=MUTED,lw=.8))
    text(a,3.4,.24,'Other nodes',10,MUTED)
    save(f,'05_contrastive_matching','对比配对矩阵','InfoNCE, van den Oord et al., 2018',{'cells':'symbolic similarity scores; diagonal is same-entity positive'})

def encoding():
    f,a=figure(8.5,2.95)
    for y,c,m in [(2.2,BLUE,'t'),(.7,ORANGE,'v')]:
        text(a,.3,y,r'$s_i^{('+m+')}$',14,c)
        arrow(a,(.65,y),(1.05,y),c)
        block(a,1.8,y,1.35,.57,'CLIP-'+('Text' if m=='t' else 'Image'),c,PALE_BLUE if m=='t' else PALE_ORANGE,10)
        arrow(a,(2.55,y),(2.95,y),c)
        vector(a,3.15,y-.25,[.7,.3,.8],c,labels=False,height=.5)
        text(a,3.5,y+.48,r'$x_i^{('+m+')}$',12,c)
        arrow(a,(4.0,y),(4.45,y),c)
        block(a,5.2,y,1.25,.57,r'$\mathrm{GNN}_{'+m+'}$',c,PALE_BLUE if m=='t' else PALE_ORANGE,12)
        arrow(a,(5.9,y),(6.4,y),c)
        vector(a,6.65,y-.25,[.5,.9,.65],c,labels=False,height=.5)
        text(a,7.0,y+.48,r'$z_i^{('+m+')}$',12,c)
    # A shared topology feeds both graph encoders.
    for u,v in [((4.9,1.45),(5.2,1.64)),((5.2,1.64),(5.48,1.3)),((4.9,1.45),(5.48,1.3))]:line(a,u,v,GREEN,.9)
    for x,y in [(4.9,1.45),(5.2,1.64),(5.48,1.3)]:node(a,x,y,'',GREEN,r=.06)
    arrow(a,(5.2,1.77),(5.2,1.89),GREEN)
    arrow(a,(5.2,1.19),(5.2,1.01),GREEN)
    text(a,5.85,1.45,r'$\mathcal{G}$',12,GREEN)
    save(f,'06_clip_graph_alignment','共享拓扑的双模态结构编码','CAME Eq. (3)-(5)',{'vector_cells':'symbolic only; parameter sets of two GNNs are separate'})
    f,a=figure(3.6,2.9)
    pos=[(.55,.55),(3.0,.55),(1.78,2.4)]
    for (u,v),s in zip([(0,1),(0,2),(1,2)],[r'$\mathcal{L}_{t,v}$',r'$\mathcal{L}_{t,f}$',r'$\mathcal{L}_{v,f}$']):
        p,q=pos[u],pos[v]
        dx,dy=q[0]-p[0],q[1]-p[1]
        mag=(dx*dx+dy*dy)**.5
        pp=(p[0]+dx/mag*.4,p[1]+dy/mag*.4)
        qq=(q[0]-dx/mag*.4,q[1]-dy/mag*.4)
        arrow(a,pp,qq,PURPLE,ls='--',both=True)
        x,y=(p[0]+q[0])/2,(p[1]+q[1])/2
        text(a,x+( -.3 if v==2 and u==0 else .3 if v==2 else 0),y+(.16 if v==1 else 0),s,11,PURPLE)
    for (x,y),s,c in zip(pos,[r'$z_i^{(t)}$',r'$z_i^{(v)}$',r'$z_i^{(f)}$'],[BLUE,ORANGE,PURPLE]):node(a,x,y,s,c,r=.33,size=13)
    save(f,'06b_tri_view_alignment','同一节点的三视图对齐','CAME Eq. (16)-(18)',{'dashed_edges':'loss constraints, not a data flow'})

def fusion():
    f,a=figure(6.8,3.3)
    # One example per row; no full-slide comparison panels.
    for yy,c,m in [(2.95,BLUE,'t'),(1.95,ORANGE,'v')]:
        text(a,.3,yy,r'$X^{('+m+')}$',13,c)
        arrow(a,(.75,yy),(1.6,2.45),c)
    node(a,1.95,2.45,r'$\mu$',PURPLE,r=.25,size=17)
    arrow(a,(2.3,2.45),(3.05,2.45))
    block(a,3.8,2.45,1.3,.56,'GNN',INK,'#F4F5F7',12)
    arrow(a,(4.52,2.45),(5.2,2.45))
    text(a,5.55,2.45,r'$Z$',14)
    for yy,c,m in [(1.4,BLUE,'t'),(.4,ORANGE,'v')]:
        text(a,.3,yy,r'$X^{('+m+')}$',13,c)
        arrow(a,(.75,yy),(1.5,yy),c)
        block(a,2.2,yy,1.3,.56,r'$\mathrm{GNN}_{'+m+'}$',c,PALE_BLUE if m=='t' else PALE_ORANGE,12)
        arrow(a,(2.93,yy),(3.45,yy),c)
        text(a,3.8,yy,r'$Z^{('+m+')}$',13,c)
        arrow(a,(4.1,yy),(4.85,.9),c)
    block(a,5.55,.9,1.25,.65,'Fusion',PURPLE,PALE_PURPLE,11)
    save(f,'07_fusion_order','平均后传播与分别传播的路径','CAME early fusion motivation',{'mu':'arithmetic mean, one instance of early fusion'})
    f,a=figure(4.7,2.6)
    for yy,left,right in [(2.0,[1,0],[0,1]),(.6,[.5,.5],[.5,.5])]:
        vector(a,.2,yy-.25,left,BLUE,width=.35,height=.6)
        vector(a,1.15,yy-.25,right,ORANGE,width=.35,height=.6)
        arrow(a,(1.98,yy+.04),(2.62,1.3),PURPLE)
    node(a,2.95,1.3,r'$\mu$',PURPLE,r=.24,size=17)
    arrow(a,(3.25,1.3),(3.65,1.3))
    vector(a,3.9,1.0,[.5,.5],PURPLE,width=.35,height=.6)
    save(f,'07b_average_noninjective','不同模态组合得到相同均值','Synthetic algebraic example',{'pairs':[[[1,0],[0,1]],[[.5,.5],[.5,.5]]],'mean':[.5,.5]})

def gates():
    f,a=figure(5.3,3.2)
    vals=[1,0]
    vector(a,.25,2.2,vals,BLUE,width=.4,height=.6)
    text(a,.65,3.0,r'$z^{(t)}$',13,BLUE)
    vector(a,.25,.25,[0,1],ORANGE,width=.4,height=.6)
    text(a,.65,.03,r'$z^{(v)}$',13,ORANGE)
    for y,c,w in [(2.5,BLUE,[.8,.2]),(.55,ORANGE,[.2,.8])]:
        arrow(a,(1.15,y),(1.6,y),c)
        node(a,1.9,y,r'$\odot$',c,r=.19,size=16)
        vector(a,1.5,1.35 if y>1 else 1.0,w,PURPLE,width=.4,height=.25)
        text(a,1.15,1.48 if y>1 else 1.12,r'$g$' if y>1 else r'$1-g$',11,PURPLE)
        if y>1:
            arrow(a,(1.9,1.68),(1.9,2.27),PURPLE)
        else:
            arrow(a,(1.9,.96),(1.9,.8),PURPLE)
        arrow(a,(2.15,y),(3.2,1.55),c)
    node(a,3.5,1.55,'+',PURPLE,r=.22,size=18)
    arrow(a,(3.8,1.55),(4.25,1.55))
    vector(a,4.4,1.25,[.8,.8],PURPLE,width=.36,height=.6)
    text(a,4.75,2.25,r'$z^{(g)}$',14,PURPLE)
    save(f,'08_gating_and_experts','逐维门控融合','CAME Eq. (6)-(7)',{'text':[1,0],'visual':[0,1],'gate':[.8,.2],'output':[.8,.8]})
    f,a=figure(6.3,2.9)
    vector(a,.1,1.2,[.4,.8,.6],BLUE,width=.2,height=.6,labels=False)
    text(a,.4,2.1,r'$z$',13,BLUE)
    arrow(a,(.8,1.5),(1.35,1.5))
    block(a,1.9,1.5,.95,.65,'Router',GREEN,'#E8F0EC',10)
    text(a,1.9,.9,'Top-2',10,GREEN)
    for y,s,on in [(2.6,r'$\phi_1$',True),(1.5,r'$\phi_2$',False),(.4,r'$\phi_3$',True)]:
        block(a,3.6,y,.9,.55,s,GREEN if on else MUTED,'#E8F0EC' if on else 'white',14,'-' if on else '--')
        if on:
            arrow(a,(2.42,1.5),(3.1,y),GREEN)
            arrow(a,(4.1,y),(5.0,1.5),GREEN)
            text(a,4.65,2.22 if y>2 else .72,'0.7' if y>2 else '0.3',10,GREEN)
    node(a,5.25,1.5,r'$\Sigma$',GREEN,r=.24,size=18)
    arrow(a,(5.55,1.5),(5.9,1.5),GREEN)
    text(a,6.1,1.5,r'$e$',14,GREEN)
    save(f,'08b_expert_routing','单个专家组内的稀疏路由','CAME intra-group routing, conceptual example',{'toy_expert_count':3,'toy_top_k':2,'weights':[.7,.3],'not_paper_hyperparameters':True})

def overview():
    sheet=Image.new('RGB',(1500,1360),'#E6E6E6' if MONOCHROME else '#E6E9ED')
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype(FONT,17)
    for i,r in enumerate(RECORDS):
        img=Image.open(OUT/(r['name']+'.png')).convert('RGBA')
        tile=Image.new('RGBA',img.size,'white')
        tile.alpha_composite(img)
        thumb=ImageOps.contain(tile.convert('RGB'),(475,280))
        x=(i%3)*500+12;y=(i//3)*340+12
        sheet.paste('white',(x,y,x+476,y+315))
        sheet.paste(thumb,(x+(476-thumb.width)//2,y+25+(280-thumb.height)//2))
        draw.text((x+8,y+6),r['concept'],font=font,fill='black' if MONOCHROME else INK)
    sheet.save((OUT if WHITE_BACKGROUND else OUT.parent)/'学术示意图总览.png')

def main():
    graph_matrix();messages();receptive_field();ssl();contrastive();encoding();fusion();gates()
    overview()
    manifest={'date':'2026-10-16','revision':('white-background-color-contrast-v1' if WHITE_BACKGROUND else 'standalone-schematics-monochrome-v1' if MONOCHROME else 'standalone-schematics-v2'),'no_experiment_data':True,
              'transparent_background':not MONOCHROME,'no_slide_layout':True,'figures':RECORDS,
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Generated',len(RECORDS),'standalone schematics, PNG + SVG.')

if __name__=='__main__':main()
