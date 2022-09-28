from re import U
from tkinter import font
from manim import *

class Channel(Scene):
    def construct(self):
        #Text description
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{mathrsfs}")
        H_space = Tex(
            r"Let $H \in \mathbb{C}^{NM \times NM}$",
            tex_template=myTemplate,
            font_size=60,
        ).shift(UP*1.5)
        
        K_description = Tex(r'which has $K_{doppler,delay}$ subblocks',font_size=60)
        
        K_space = Tex(
            r"where $K_{m,l} \in \mathbb{C}^{N \times N}$",
            tex_template=myTemplate,
            font_size=60,
        ).shift(DOWN*1.5)
        
        description = VGroup()
        description.add(H_space)
        description.add(K_description)
        description.add(K_space)
        
        ##Second animation
       
        H_matrix = MobjectMatrix([
            [Tex(r'$K_{0,0}$'),Square().scale(0.3),Tex(r'$K_{0,2}$'),Tex(r'$K_{0,1}$')],
            [Tex(r'$K_{1,1}$'),Tex(r'$K_{1,0}$'),Square().scale(0.3),Tex(r'$K_{1,2}$')],
            [Tex(r'$K_{2,2}$'),Tex(r'$K_{2,1}$'),Tex(r'$K_{2,0}$'),Square().scale(0.3)],
            [Square().scale(0.3),Tex(r'$K_{3,2}$'),Tex(r'$K_{3,1}$'),Tex(r'$K_{3,0}$')]
        ])
        bra = H_matrix.get_brackets()
        ent = H_matrix.get_entries()
        ent[0].set_color(BLUE)
        ent[3].set_color(BLUE)
        ent[2].set_color(BLUE)
        
        ent[5].set_color(GREEN)
        ent[4].set_color(GREEN)
        ent[7].set_color(GREEN)
        
        ent[8].set_color(PINK)
        ent[9].set_color(PINK)
        ent[10].set_color(PINK)
        
        ent[13].set_color(PURE_GREEN)
        ent[14].set_color(PURE_GREEN)
        ent[15].set_color(PURE_GREEN)
        
        
        arrow_up = DoubleArrow(start=ent[0].get_bottom(), end=ent[0].get_top(), color=GOLD).shift(LEFT*.55)
        size_k_l  = Tex("N",font_size=20)
        size_k_l.set_x(arrow_up.get_x())
        size_k_l.set_y(arrow_up.get_y())
        size_k_l.shift(LEFT*.2)
        
        arrow_lr = DoubleArrow(start=ent[0].get_left(), end=ent[0].get_right(), color=GOLD).shift(UP*.55)
        size_k_b = Tex("N",font_size=20)
        size_k_b.set_x(arrow_lr.get_x())
        size_k_b.set_y(arrow_lr.get_y())
        size_k_b.shift(UP*.2)
        
        size_k_group = VGroup()
        size_k_group.add(arrow_up)
        size_k_group.add(size_k_l)
        size_k_group.add(arrow_lr)
        size_k_group.add(size_k_b)
        
        y_reduced = MobjectMatrix([
            [Tex(r'$y_0$')],
            [Tex(r'$y_1$')],
            [Tex(r'$y_2$')],
            [Tex(r'$y_3$')]
        ]).shift(LEFT*4.5)
        eq = Tex(" = ",font_size=60).shift(LEFT*3.5)
        
        
        x_reduced = MobjectMatrix([
            [Tex(r'$x_0$')],
            [Tex(r'$x_1$')],
            [Tex(r'$x_2$')],
            [Tex(r'$x_3$')]
        ]).shift(RIGHT*4)
        
        #self.play(Create(description))
        #self.wait(6)
        #self.play(Unwrite(description))
        #self.wait()
        
        self.play(Create(bra))
        self.play(Create(size_k_group))
        for i in range(0,4): 
            for j in range(0,4):
                self.play(Create(ent[((i-j)%4)+(i*4)]))
                self.wait(1)
                
        self.play(Create(y_reduced))
        self.play(Create(eq))
        self.play(Create(x_reduced))
        self.wait(1)
        
        