"""Premium GUI for Network Routing Simulation using customtkinter."""
import customtkinter as ctk
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading, queue, math
import networkx as nx
import numpy as np
from q_learning_agent import QLearningAgent, compute_reward, extract_path
from dijkstra_routing import format_path

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
R = 26  # node radius
PAL = {"canvas":"#0f1120","grid":"#191b30","node":"#06b6d4","node_b":"#0891b2",
       "src":"#10b981","src_b":"#059669","tgt":"#f59e0b","tgt_b":"#d97706",
       "edge":"#475569","wt":"#fb7185","dijk":"#3b82f6","ql":"#ef4444",
       "cong":"#f97316","sel":"#a855f7","glow1":"#0e293a","glow2":"#0c3547"}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Network Routing Simulator")
        self.geometry("1350x780")
        self.minsize(1100,650)
        self.nodes={}; self.edges={}; self.nid=0
        self.source=self.target=None; self.mode="node"
        self.estart=None; self.drag=None; self.agent=None
        self.mq=queue.Queue(); self.busy=False
        self.dp=self.dc=self.qp=self.qc=None; self.cong=set()
        self._ui()

    def _ui(self):
        # Header
        hdr=ctk.CTkFrame(self,height=50,corner_radius=0,fg_color="#1a1b2e")
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr,text="  Network Routing Simulator",
                     font=("Segoe UI",18,"bold"),text_color="#22d3ee").pack(side="left",padx=12)
        ctk.CTkLabel(hdr,text="Dijkstra vs Q-Learning  ",
                     font=("Segoe UI",12),text_color="#64748b").pack(side="right",padx=12)
        body=ctk.CTkFrame(self,fg_color="transparent")
        body.pack(fill="both",expand=True,padx=10,pady=(4,10))
        # Sidebar
        sb=ctk.CTkScrollableFrame(body,width=260,corner_radius=12,fg_color="#1e2036")
        sb.pack(side="left",fill="y",padx=(0,8))
        self._sidebar(sb)
        # Right
        rt=ctk.CTkFrame(body,fg_color="transparent")
        rt.pack(side="left",fill="both",expand=True)
        # Canvas
        cf=ctk.CTkFrame(rt,corner_radius=12,fg_color="#1a1b2e")
        cf.pack(fill="both",expand=True,pady=(0,6))
        self.cv=tk.Canvas(cf,bg=PAL["canvas"],highlightthickness=0,cursor="crosshair")
        self.cv.pack(fill="both",expand=True,padx=3,pady=3)
        self.cv.bind("<Button-1>",self._click)
        self.cv.bind("<B1-Motion>",self._drag_ev)
        self.cv.bind("<ButtonRelease-1>",lambda e:setattr(self,'drag',None))
        self.cv.bind("<Button-3>",self._rclick)
        self.cv.bind("<Configure>",lambda e:self._draw())
        self.stv=ctk.StringVar(value="  Add Node — click on canvas to place routers")
        ctk.CTkLabel(cf,textvariable=self.stv,font=("Segoe UI",11),
                     text_color="#64748b",anchor="w").pack(fill="x",padx=10,pady=(0,6))
        # Results tabs
        self.tabs=ctk.CTkTabview(rt,height=180,corner_radius=10,
                                  fg_color="#1e2036",segmented_button_fg_color="#2d2f4e",
                                  segmented_button_selected_color="#7c3aed")
        self.tabs.pack(fill="x")
        self.tabs.add("Log"); self.tabs.add("Comparison")
        self.log=ctk.CTkTextbox(self.tabs.tab("Log"),font=("Consolas",11),
                                fg_color="#141524",text_color="#e2e8f0",corner_radius=8)
        self.log.pack(fill="both",expand=True,padx=4,pady=4)
        self.cmp=ctk.CTkTextbox(self.tabs.tab("Comparison"),font=("Consolas",11),
                                fg_color="#141524",text_color="#e2e8f0",corner_radius=8)
        self.cmp.pack(fill="both",expand=True,padx=4,pady=4)

    def _sidebar(self,sb):
        # Mode
        ctk.CTkLabel(sb,text="MODE",font=("Segoe UI",11,"bold"),text_color="#22d3ee").pack(anchor="w",pady=(8,4))
        self.seg=ctk.CTkSegmentedButton(sb,values=["Node","Edge","Source","Dest"],
                                         command=self._mode_cb,font=("Segoe UI",11),
                                         selected_color="#7c3aed",selected_hover_color="#6d28d9")
        self.seg.set("Node"); self.seg.pack(fill="x",pady=(0,8))
        sep(sb)
        # Actions
        ctk.CTkLabel(sb,text="ACTIONS",font=("Segoe UI",11,"bold"),text_color="#22d3ee").pack(anchor="w",pady=(8,4))
        ctk.CTkButton(sb,text="Run Dijkstra",fg_color="#2563eb",hover_color="#1d4ed8",
                      command=self.run_dijk,font=("Segoe UI",12,"bold"),height=36).pack(fill="x",pady=3)
        ctk.CTkButton(sb,text="Train Q-Learning",fg_color="#059669",hover_color="#047857",
                      command=self.run_ql,font=("Segoe UI",12,"bold"),height=36).pack(fill="x",pady=3)
        ef=ctk.CTkFrame(sb,fg_color="transparent")
        ef.pack(fill="x",pady=2)
        ctk.CTkLabel(ef,text="Episodes:",font=("Segoe UI",10),text_color="#94a3b8").pack(side="left")
        self.ep_var=ctk.StringVar(value="1000")
        ctk.CTkEntry(ef,textvariable=self.ep_var,width=70,height=28,
                     font=("Segoe UI",11),corner_radius=6).pack(side="right")
        self.prog=ctk.CTkProgressBar(sb,height=6,corner_radius=3,
                                      progress_color="#22d3ee",fg_color="#2d2f4e")
        self.prog.set(0); self.prog.pack(fill="x",pady=(4,8))
        sep(sb)
        # Quick
        ctk.CTkLabel(sb,text="QUICK",font=("Segoe UI",11,"bold"),text_color="#22d3ee").pack(anchor="w",pady=(8,4))
        ctk.CTkButton(sb,text="Load Example Graph",fg_color="#d97706",hover_color="#b45309",
                      command=self.load_ex,font=("Segoe UI",11,"bold"),height=32).pack(fill="x",pady=2)
        bf=ctk.CTkFrame(sb,fg_color="transparent")
        bf.pack(fill="x",pady=2)
        ctk.CTkButton(bf,text="Clear Paths",fg_color="#334155",hover_color="#475569",
                      command=self.clr_paths,height=30,font=("Segoe UI",10)).pack(side="left",fill="x",expand=True,padx=(0,2))
        ctk.CTkButton(bf,text="Clear All",fg_color="#991b1b",hover_color="#7f1d1d",
                      command=self.clr_all,height=30,font=("Segoe UI",10)).pack(side="left",fill="x",expand=True,padx=(2,0))
        sep(sb)
        # Congestion
        ctk.CTkLabel(sb,text="CONGESTION",font=("Segoe UI",11,"bold"),text_color="#fb923c").pack(anchor="w",pady=(8,4))
        self.ecb=ctk.CTkComboBox(sb,state="readonly",font=("Segoe UI",10),
                                  dropdown_fg_color="#2d2f4e",corner_radius=6,height=30)
        self.ecb.pack(fill="x",pady=2)
        wf=ctk.CTkFrame(sb,fg_color="transparent")
        wf.pack(fill="x",pady=2)
        ctk.CTkLabel(wf,text="New Weight:",font=("Segoe UI",10),text_color="#94a3b8").pack(side="left")
        self.wv=ctk.StringVar(value="20")
        ctk.CTkEntry(wf,textvariable=self.wv,width=60,height=28,corner_radius=6).pack(side="right")
        ctk.CTkButton(sb,text="Apply Congestion",fg_color="#ea580c",hover_color="#c2410c",
                      command=self.do_cong,height=32,font=("Segoe UI",11,"bold")).pack(fill="x",pady=(4,8))

    # ── Helpers ──
    def _mode_cb(self,v):
        m={"Node":"node","Edge":"edge","Source":"source","Dest":"dest"}[v]
        self.mode=m; self.estart=None
        tips={"node":"  Click canvas to place routers","edge":"  Click two nodes to connect",
              "source":"  Click a node → SOURCE","dest":"  Click a node → DESTINATION"}
        self.stv.set(tips[m])

    def _msg(self,t):
        self.log.insert("end",t+"\n"); self.log.see("end")

    def _hit(self,x,y):
        for i,(nx_,ny) in self.nodes.items():
            if math.hypot(x-nx_,y-ny)<=R+5: return i
        return None

    def _combo(self):
        vals=[f"{u} -> {v}  (w={w})" for (u,v),w in self.edges.items()]
        self.ecb.configure(values=vals)
        if vals: self.ecb.set(vals[0])

    def _G(self):
        G=nx.DiGraph(); G.add_nodes_from(self.nodes.keys())
        for (u,v),w in self.edges.items(): G.add_edge(u,v,weight=w)
        return G

    def _ok(self):
        if self.source is None or self.target is None:
            messagebox.showwarning("","Set Source and Destination first."); return False
        if not self.edges:
            messagebox.showwarning("","Add edges first."); return False
        return True

    # ── Drawing ──
    def _draw(self):
        c=self.cv; c.delete("all")
        w,h=c.winfo_width(),c.winfo_height()
        for x in range(0,w,50): c.create_line(x,0,x,h,fill=PAL["grid"])
        for y in range(0,h,50): c.create_line(0,y,w,y,fill=PAL["grid"])
        for (u,v),wt in self.edges.items(): self._de(u,v,wt,PAL["cong"] if (u,v) in self.cong else PAL["edge"])
        if self.dp: self._dpath(self.dp,PAL["dijk"],-7)
        if self.qp: self._dpath(self.qp,PAL["ql"],7)
        for i,(x,y) in self.nodes.items(): self._dn(i,x,y)
        self._legend()

    def _dn(self,i,x,y):
        c=self.cv; s=i==self.source; t=i==self.target; sel=i==self.estart
        fl=PAL["src"] if s else PAL["tgt"] if t else PAL["sel"] if sel else PAL["node"]
        bd=PAL["src_b"] if s else PAL["tgt_b"] if t else "#7c3aed" if sel else PAL["node_b"]
        for dr,gc in [(14,PAL["glow1"]),(10,PAL["glow2"]),(6,fl)]:
            r=R+dr; c.create_oval(x-r,y-r,x+r,y+r,fill="",outline=gc,width=2)
        c.create_oval(x-R,y-R,x+R,y+R,fill=fl,outline=bd,width=3)
        c.create_text(x,y,text=str(i),font=("Segoe UI",14,"bold"),fill="#0f1120")
        if s: c.create_text(x,y+R+15,text="SRC",font=("Segoe UI",8,"bold"),fill=PAL["src"])
        if t: c.create_text(x,y+R+15,text="DEST",font=("Segoe UI",8,"bold"),fill=PAL["tgt"])

    def _ep(self,u,v,off=0):
        x1,y1=self.nodes[u]; x2,y2=self.nodes[v]
        dx,dy=x2-x1,y2-y1; d=math.hypot(dx,dy)
        if d==0: return x1,y1,x2,y2
        ux,uy=dx/d,dy/d; px,py=-uy*off,ux*off
        return x1+ux*R+px,y1+uy*R+py,x2-ux*R+px,y2-uy*R+py

    def _de(self,u,v,w,col):
        off=8 if (v,u) in self.edges else 0
        sx,sy,ex,ey=self._ep(u,v,off)
        self.cv.create_line(sx,sy,ex,ey,arrow=tk.LAST,arrowshape=(10,13,5),fill=col,width=2,smooth=True)
        mx,my=(sx+ex)/2,(sy+ey)/2
        dx,dy=ex-sx,ey-sy; d=max(math.hypot(dx,dy),1)
        self.cv.create_text(mx-dy/d*15,my+dx/d*15,text=str(w),font=("Segoe UI",9,"bold"),fill=PAL["wt"])

    def _dpath(self,path,col,off):
        for i in range(len(path)-1):
            u,v=path[i],path[i+1]; bo=8 if (v,u) in self.edges else 0
            sx,sy,ex,ey=self._ep(u,v,bo+off)
            da=(10,5) if col==PAL["ql"] else ()
            self.cv.create_line(sx,sy,ex,ey,arrow=tk.LAST,arrowshape=(14,17,7),fill=col,width=5,dash=da,smooth=True)

    def _legend(self):
        if not self.dp and not self.qp: return
        x,y=20,20; c=self.cv
        c.create_rectangle(x-8,y-10,x+130,y+(40 if self.dp and self.qp else 16),
                           fill="#1e2036",outline="#3d3f66",width=1)
        if self.dp:
            c.create_line(x,y,x+28,y,fill=PAL["dijk"],width=3)
            c.create_text(x+34,y,text=f"Dijkstra (cost:{self.dc})",anchor="w",font=("Segoe UI",9,"bold"),fill=PAL["dijk"])
            y+=22
        if self.qp:
            c.create_line(x,y,x+28,y,fill=PAL["ql"],width=3,dash=(6,3))
            c.create_text(x+34,y,text=f"Q-Learn (cost:{self.qc})",anchor="w",font=("Segoe UI",9,"bold"),fill=PAL["ql"])

    # ── Interaction ──
    def _click(self,e):
        if self.busy: return
        h=self._hit(e.x,e.y)
        if self.mode=="node":
            if h is None:
                self.nodes[self.nid]=(e.x,e.y)
                self.stv.set(f"  Added Node {self.nid}"); self.nid+=1; self._draw()
        elif self.mode=="edge":
            if h is not None:
                if self.estart is None:
                    self.estart=h; self.stv.set(f"  Edge from {h} → click target"); self._draw()
                else:
                    if h!=self.estart: self._pedge(self.estart,h)
                    self.estart=None; self._draw()
        elif self.mode=="source":
            if h is not None: self.source=h; self.stv.set(f"  Source = {h}"); self._draw()
        elif self.mode=="dest":
            if h is not None: self.target=h; self.stv.set(f"  Dest = {h}"); self._draw()

    def _drag_ev(self,e):
        if self.busy: return
        if self.drag is None: self.drag=self._hit(e.x,e.y)
        if self.drag is not None and self.mode=="node":
            self.nodes[self.drag]=(e.x,e.y); self._draw()

    def _rclick(self,e):
        if self.busy: return
        h=self._hit(e.x,e.y)
        m=tk.Menu(self,tearoff=0,bg="#2d2f4e",fg="#e2e8f0",activebackground="#7c3aed",
                  activeforeground="#fff",font=("Segoe UI",10))
        if h is not None:
            m.add_command(label=f"Delete Node {h}",command=lambda:self._deln(h))
        else:
            for (u,v) in list(self.edges):
                sx,sy,ex,ey=self._ep(u,v)
                if math.hypot(e.x-(sx+ex)/2,e.y-(sy+ey)/2)<22:
                    m.add_command(label=f"Delete {u}->{v}",command=lambda a=u,b=v:self._dele(a,b)); break
        if m.index("end") is not None: m.tk_popup(e.x_root,e.y_root)

    def _pedge(self,u,v):
        w=simpledialog.askinteger("Weight",f"Weight for {u} → {v}:",minvalue=1,maxvalue=100,initialvalue=5,parent=self)
        if w is None: return
        bi=messagebox.askyesno("Direction",f"Bidirectional ({u} ↔ {v})?",parent=self)
        self.edges[(u,v)]=w
        if bi: self.edges[(v,u)]=w
        self._combo()

    def _deln(self,n):
        del self.nodes[n]
        self.edges={k:v for k,v in self.edges.items() if n not in k}
        if self.source==n: self.source=None
        if self.target==n: self.target=None
        self._combo(); self.clr_paths()

    def _dele(self,u,v):
        self.edges.pop((u,v),None); self.cong.discard((u,v)); self._combo(); self.clr_paths()

    # ── Algorithms ──
    def run_dijk(self):
        if not self._ok(): return
        G=self._G()
        try:
            p=nx.dijkstra_path(G,self.source,self.target,weight="weight")
            c=int(nx.dijkstra_path_length(G,self.source,self.target,weight="weight"))
        except nx.NetworkXNoPath:
            self._msg("[!] No path exists!"); return
        self.dp,self.dc=p,c; self._msg(f"[Dijkstra] {format_path(p)}, Cost: {c}")
        self._table(); self._draw()

    def run_ql(self):
        if not self._ok(): return
        try: eps=int(self.ep_var.get())
        except: messagebox.showerror("","Episodes must be a number."); return
        self.busy=True; self.prog.set(0); self.stv.set("  Training Q-Learning ...")
        self.agent=QLearningAgent(self.nid,self.nid); G=self._G()
        s,t=self.source,self.target
        def work():
            a=self.agent
            for ep in range(1,eps+1):
                st,vis,tot=s,{s},0.0
                for _ in range(50):
                    nb=list(G.neighbors(st))
                    if not nb: break
                    act=a.choose_action(st,nb)
                    rw=compute_reward(G,st,act,t,vis)
                    a.update(st,act,rw,act); tot+=rw; vis.add(act); st=act
                    if st==t: break
                a.epsilon=max(0.05,a.epsilon*0.995)
                if ep%100==0: self.mq.put(("p",ep/eps))
                if ep%200==0: self.mq.put(("l",f"  Ep {ep}: Reward={tot:.1f}"))
            p2,c2=extract_path(a,s,t,G); self.mq.put(("d",(p2,c2)))
        threading.Thread(target=work,daemon=True).start(); self._poll()

    def _poll(self):
        while not self.mq.empty():
            k,d=self.mq.get_nowait()
            if k=="l": self._msg(d)
            elif k=="p": self.prog.set(d)
            elif k=="d":
                self.qp,self.qc=d; self.prog.set(1)
                self._msg(f"[Q-Learning] {format_path(self.qp)}, Cost: {self.qc}")
                self.busy=False; self.stv.set("  Training complete!")
                self._table(); self._draw(); return
        if self.busy: self.after(80,self._poll)

    def do_cong(self):
        sel=self.ecb.get()
        if not sel: messagebox.showwarning("","Select an edge."); return
        try: nw=int(self.wv.get())
        except: messagebox.showerror("","Weight must be a number."); return
        p=sel.split(); u,v=int(p[0]),int(p[2])
        old=self.edges[(u,v)]; self.edges[(u,v)]=nw; self.cong.add((u,v))
        self._msg(f"[Congestion] {u}->{v}: {old} -> {nw}"); self._combo(); self.clr_paths()

    def _table(self):
        if not self.dp or not self.qp: return
        self.cmp.delete("1.0","end")
        self.cmp.insert("end","  COMPARISON TABLE\n")
        self.cmp.insert("end","  "+"-"*46+"\n")
        self.cmp.insert("end",f"  {'Strategy':<14} {'Path':<18} {'Cost':>5} {'Hops':>5}\n")
        self.cmp.insert("end","  "+"-"*46+"\n")
        self.cmp.insert("end",f"  {'Dijkstra':<14} {format_path(self.dp):<18} {self.dc:>5} {len(self.dp)-1:>5}\n")
        self.cmp.insert("end",f"  {'Q-Learning':<14} {format_path(self.qp):<18} {self.qc:>5} {len(self.qp)-1:>5}\n")
        self.cmp.insert("end","  "+"-"*46+"\n")
        diff=self.qc-self.dc
        if diff==0: self.cmp.insert("end","  Both found the SAME optimal path!\n")
        elif diff>0: self.cmp.insert("end",f"  Dijkstra is cheaper by {diff}\n")
        else: self.cmp.insert("end",f"  Q-Learning is cheaper by {-diff}\n")
        self.tabs.set("Comparison")

    def clr_paths(self):
        self.dp=self.dc=self.qp=self.qc=None; self._draw()

    def clr_all(self):
        self.nodes.clear(); self.edges.clear(); self.cong.clear()
        self.nid=0; self.source=self.target=None
        self.dp=self.dc=self.qp=self.qc=None; self.agent=self.estart=None
        self.log.delete("1.0","end"); self.cmp.delete("1.0","end")
        self.prog.set(0); self._combo(); self._draw()

    def load_ex(self):
        self.clr_all()
        self.after(50,self._do_load)

    def _do_load(self):
        w=max(self.cv.winfo_width(),700); h=max(self.cv.winfo_height(),450)
        for i,(fx,fy) in {0:(.12,.78),1:(.40,.10),2:(.12,.38),3:(.40,.42),4:(.40,.78),5:(.72,.42)}.items():
            self.nodes[i]=(w*fx,h*fy)
        self.nid=6
        for u,v,wt in [(0,4,3),(4,0,3),(4,3,2),(3,4,2),(3,1,3),(1,3,3),(3,2,4),(2,3,4),(1,5,8),(4,5,5),(5,1,1),(5,4,1)]:
            self.edges[(u,v)]=wt
        self.source,self.target=2,5; self._combo()
        self._msg("Loaded 6-node example. Source=2, Dest=5"); self._draw()


def sep(p):
    ctk.CTkFrame(p,height=2,fg_color="#3d3f66",corner_radius=0).pack(fill="x",pady=6)

if __name__=="__main__":
    App().mainloop()
