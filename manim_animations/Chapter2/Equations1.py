from re import U, X
from turtle import left
from manim import *

class Channel(Scene):
    def construct(self):
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage[dvipsnames]{xcolor}")
        sampled = Tex(r'Time sampled $t=\frac{qT}{M}$ and doppler $\tau=\frac{lT}{M}$',font_size=60)
        g1 = Tex(r'$ g^s[l,q]=\sum _{l\in L}\sum _{k\in K} v_{l}( k) e^{j2 \pi k_i \frac{\Delta f}{N}\left( t-{\frac{lT}{M}}\right)}$',
                font_size=60)
        #g1[0][10:12].set_color(RED)
        #g1[0][42:46].set_color(RED)
        #g1[0][33:39].set_color(BLUE)
        g2 = Tex(r'$g(\tau_i,t)$',font_size=60).shift(LEFT*4.8)
        g3 = Tex(r'$g(\tau_i = \frac{lT}{M},t=\frac{qT}{M})$',font_size=60).set_x(g1[0][1:7].get_left()[0]+1)
        g3.shift(UP*2)
        exp1 = Tex(r'$\frac{qT}{M}$',font_size=45).set_x(g1[0][33].get_left()[0]).shift(UP*.16)
        exp2 = Tex(r'$\frac{\Delta f}{N}$',font_size=45).set_x(g1[0][33].get_left()[0]).shift(UP*.16)
        exp2.shift(RIGHT*1)
        exp3 = Tex(r'$(\frac{q}{NM}-\frac{l}{NM})$',font_size=45).set_x(g1[0][33].get_left()[0]).shift(UP*.16)
        exp4 = Tex(r'$e^{\frac{j 2 \pi k_i}{NM}(q-l)}$',font_size=60).set_x(g1[0][22].get_right()[0]).shift(RIGHT*1)
        exp5 = Tex(r'$z^{k_i(q-l)}$',font_size=60).set_x(g1[0][22].get_right()[0]).shift(RIGHT*.5)
        
        info =  Tex(r'$\Delta f*T=1$',font_size=60).shift(UP*2)
        info.shift(RIGHT)
        info2 = Tex(r'$z = e^{\frac{2*\pi*j}{NM}}$',font_size=60).shift(UP*2)
        info2.shift(RIGHT)
        
        self.play(Create(g1))
        self.wait(2)
        self.play(ReplacementTransform(g1[0][0:7], g2))
        self.wait(3)
        self.play(ReplacementTransform(g2, g3))
        self.wait(3)
        self.play(ChangeSpeed( AnimationGroup(g1[0][33:].animate(runtime=1).shift(RIGHT*.3)),speedinfo={1: 1}))
        self.play(ReplacementTransform(g1[0][32:33], exp1))
        self.play(ChangeSpeed( AnimationGroup(g1[0][27:31].animate(runtime=1).shift(RIGHT*.2),
                                              g1[0][31].animate(runtime=1).shift(LEFT*.7), #parenthesis
                                              g1[0][34:].animate(runtime=1).shift(RIGHT*.7)
                                              ),speedinfo={1: 1}))
        self.play(Create(info))
        
        self.play(Create(exp2))
        self.wait(2)    
        
        vg  = VGroup()
        vg.add(exp1)
        vg.add(exp2)
        vg.add(g1[0][27:])
        
        self.play(ReplacementTransform(vg,exp3))
        self.wait(3)
        vg2  = VGroup()
        vg2.add(g1[0][21:31])
        vg2.add(exp3)
        
        self.play(ReplacementTransform(vg2,exp4))
        self.play(ReplacementTransform(info,info2))
        self.wait(3)
        self.play(ReplacementTransform(exp4,exp5))
        self.wait(3)
        