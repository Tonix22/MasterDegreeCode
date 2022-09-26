from re import U
from manim import *


class AnimatedSquareToCircle(Scene):
    def construct(self):
        #First definition matrix
        Y = Tex("Y = ").shift(4 * LEFT)
        y = Tex("y = ").shift(2.5 * LEFT)

        
        m0 = MobjectMatrix([
                            [Tex(r'$Y_{\tau_0 \nu_0}$'),Tex(r'$\cdots$'),Tex(r'$Y_{\tau_0 \nu_N}$')], 
                            [Tex(r'$\vdots$'),Tex(r'$\ddots$'),Tex(r'$\vdots$')],
                            [Tex(r'$Y_{\tau_M \nu_0}$'),Tex(r'$\cdots$'),Tex(r'$Y_{\tau_M \nu_N}$')],
                        ])
        
        m0_rows = m0.get_rows()
        m0.add(SurroundingRectangle(m0_rows[0]))
        m0.add(SurroundingRectangle(m0_rows[2],color='GREEN'))
        eq_m0 = VGroup()
        delay_arrow   = Tex(r"$\downarrow{\tau = delay}$", font_size=40).shift(3.5*RIGHT)
        doppler_arrow = Tex(r"$\rightarrow{\nu = doppler}$", font_size=40).shift(2*DOWN)
        eq_m0.add(Y,m0,delay_arrow,doppler_arrow)
        
        
        #vectorized verbose
        v1 = MobjectMatrix([
                     [Tex(r'$Y_{\tau_0 \nu_0}$')],
                     [Tex(r'$\vdots$')], 
                     [Tex(r'$Y_{\tau_0 \nu_N}$')], 
                     [Tex(r'$Y_{\tau_1 \nu_0}$')],
                     [Tex(r'$\vdots$')],
                     [Tex(r'$Y_{\tau_1 \nu_N}$')],
                     [Tex(r'$\vdots$')],
                     [Tex(r'$Y_{\tau_M \nu_N}$')]
                     ])
        #Get on block of vectorized form
        ent = v1.get_entries()
        vg  = VGroup()
        vg.add(ent[0])
        vg.add(ent[1])
        vg.add(ent[2])
        v1.add(SurroundingRectangle(vg))
        
        #
        vg2 = VGroup()
        vg2.add(ent[3])
        vg2.add(ent[4])
        vg2.add(ent[5])
        v1.add(SurroundingRectangle(vg2,color='GREEN'))
        
        eq_v1 = VGroup()
        tap_delay1 = Tex(r"$\rightarrow{delayTap1}$", font_size=40).shift(2.5*RIGHT).shift(2*UP)
        tap_delay2 = Tex(r"$\rightarrow{delayTap2}$", font_size=40).shift(2.5*RIGHT).shift(.3*DOWN)
        tap_delayN = Tex(r"$\rightarrow{delayTapM}$", font_size=40).shift(2.5*RIGHT).shift(3*DOWN)
        
        eq_v1.add(y,v1,tap_delay1,tap_delay2,tap_delayN)
        
        #Vectorized reduced
        v_reduced = MobjectMatrix([
            [Tex(r'$y_0$')],
            [Tex(r'$y_1$')],
            [Tex(r'$\vdots$')],
            [Tex(r'$y_{M-1}$')]
        ])
        v_reduced.add(SurroundingRectangle(v_reduced.get_rows()[0]))
        v_reduced.add(SurroundingRectangle(v_reduced.get_rows()[1],color='GREEN'))
        eq_v_reduced = VGroup()
        eq_v_reduced.add(y,v_reduced)
        
        #fourth animation
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{mathrsfs}")
        
        m_space = Tex(
            r"$m \in \{0,\cdots,M-1\}$",
            tex_template=myTemplate,
            font_size=40,
        ).shift(3*RIGHT).shift(1.5*UP).shift(RIGHT)
        
        y_m_space = Tex(
            r"$y_m \in \mathbb{C}^{N \times 1}$",
            tex_template=myTemplate,
            font_size=40,
        ).shift(3*RIGHT)
        
        spaces =  VGroup()
        spaces.add(m_space,y_m_space)
        
    
        self.play(Create(eq_m0))  # animate the creation of the square
        self.wait(5)
        self.play(ReplacementTransform(eq_m0, eq_v1))  # interpolate the square into the circle
        self.wait(5)
        self.play(ReplacementTransform(eq_v1, eq_v_reduced))
        self.wait(2)
        self.play(Create(spaces))
        self.wait(5)
        self.play(Unwrite(eq_v_reduced))
        self.play(Unwrite(spaces))
        
