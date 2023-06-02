import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time
#from PIL import Image
######################################### 
###########################################################
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

g = 9.81

######################################### RELATION FORCE/COMPRESSION ###########################################################
def F(DX, EI, Lp):
    return np.pi**2 * EI/(Lp**2) * (1 + 0.45 * DX/Lp)


####################################### CONDITIONS SUR LA RAIDEUR#########################################################
# La perche ne doit pas flamber sous le poids propre :
def min_EI(m, Lp):
    return m * g * (Lp**2) / (np.pi**2)

# Le perchiste doit être capable de plier la perche avec sa course d'élan
def max_EI(m, Lp, v0, k):
    return np.sqrt(k*m) * v0 * (Lp**2) / (np.pi**2)


############################################# Perche déformée ##############################################################

# on trace un arc de cercle corrspondant respectant le déplacement Dx de l'extrémité de la perche

def f_alpha(alpha, *args):
    '''
    l'angle alpha de l'arc de cercle vérifie f_alpha(alpha) = 0
    '''
    Lp, L = args
    return np.sin(alpha/2) - L/(2*Lp)*alpha

def perche_deformee(x, y, Lp):
    '''
    renvoie la liste X, Y des points correspondant à une déformée de la perche.
    Une extrémité de la perche est en (0, 0), l'autre est en (x, y)
    Lp est la longueur de la perche non déformée
    '''
    if np.sqrt(x**2 + y**2) >= Lp:
        phi = np.arctan2(y, x)
        Xp = np.array([0, Lp*np.cos(phi)])
        Yp = np.array([0, Lp*np.sin(phi)])
    else:
        L = np.sqrt(x**2 + y**2)
        alpha0 = np.sqrt(48*(0.5 - 0.5*L/Lp)) # initial guess
        alpha = fsolve(f_alpha, alpha0, args=(Lp, L))[0]
        r = Lp/alpha
        # déformée sur l'axe horizontal
        theta = np.linspace(np.pi/2 - alpha/2, np.pi/2 + alpha/2, 100)
        X = r * np.cos(theta) + L/2
        Y = r * np.sin(theta) - r*np.cos(alpha/2)
        # rotation de la déformée pour que l'extrémité soit en x, y
        phi = np.arctan2(y, x)
        mat_R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        Xp, Yp = np.dot(mat_R, np.array([X, Y]))
    return Xp, Yp

############################################ Trajectoire du sauteur ####################################################################
def linearPassiveVault(t,y,K,L):
    x,z,ux,uz = y
    R = (x**2+z**2)**0.5
    duxdt = K*(1+0.45*(L-R)/L)*x/R
    duzdt = K*(1+0.45*(L-R)/L)*z/R-1
    dxdt = ux
    dzdt = uz
    dydt = [dxdt,dzdt,duxdt,duzdt]
    return dydt
def hit_ground(t,y,K,L): return y[1]
hit_ground.terminal=True
hit_ground.direction=-1
def recoil(t,y,K,L): return ((y[0]**2+y[1]**2)-L**2)
recoil.terminal=True
recoil.direction=1
def flightDyn(t,y,K,L):
    x,z,ux,uz = y
    duxdt = 0
    duzdt = -1
    dxdt = ux
    dzdt = uz
    dydt = [dxdt,dzdt,duxdt,duzdt]
    return dydt
def maxHeight(t,y,K,L): return (y[3])


def classifyJump(K,L,H0,v0,dt=0.1,dt_max=0.2):
    '''
    classifie un saut:
    0 : tombe sur la piste (trop rigide)
    1 : renvoyé vers la piste (trop rigide)
    2 : perche molle
    3 : bon saut
    -1 : erreur
    '''
    #initial conditions:
    y0=[(L**2-1)**0.5,1,-v0/np.sqrt(g*H0),0]
    #solve 
    sol=solve_ivp(linearPassiveVault,[0,5],y0,events=(recoil,hit_ground),args=(K,L,),first_step=dt,max_step=dt_max)
    ytoff = sol.y[:,-1]
    sol_free=solve_ivp(linearPassiveVault,[0,5],y0,events=(recoil,hit_ground),args=(0,L,),first_step=dt,max_step=dt_max)
    sol_flight = sol_free #default
    res=[0,0]
    if len(sol.t_events[1])>0:
      #touche le sol
      y_flag = 0
    else:
      #autre cas
      if len(sol.t_events[0])>0:
        #la perche se retend
        if ytoff[2]>0:
          #perche repart vers la piste
          y_flag=1
        else:
          #il passe une barre et on note la hauteur
          y_flag=2
          #solve the trajectory
          sol_flight = solve_ivp(flightDyn,[0,10],ytoff,events=(maxHeight,hit_ground),args=(K,L*H0,),first_step=dt,max_step=dt_max)
          i_apex = np.argmax(sol_flight.y[1,:])
          h_apex = sol_flight.y[1,i_apex]
          x_apex = sol_flight.y[0,i_apex]
          res = [x_apex,h_apex]
          if h_apex>L:
            y_flag=3 #good jump
      else:
        y_flag = -1
    return y_flag, sol, sol_flight

################################### Main #################################
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
import matplotlib.animation as animation



############################################### NOTATIONS #####################################################################
# m : masse du perchiste
# H0 : hauteur initiale du centre de gravité du sauteur
# Lp : taille de la perche
# L = Lp/H0
# EI : raideur de la perche
# F : force appliquée par le perchiste sur la perche
# DX : pour l'expérience de compression de la perche (verticale) : Lp - hauteur de la perche déformée

################################################ CONSTANTES ####################################################################
g = 9.81
k = 250 # raideur des épaules et du tronc (N/m) : Gros et Kunkel 1998
H = 6 # fixé arbitrairement cf Jupiter_Remi
Fr = (2*(H-1))**0.5 # cf Jupiter Remi
WR = 6.22 # record du monde détenu par Duplantis

##################################################### SAUT ####################################################################
message = ["Touche le sol !", "Repart vers la piste !", "Saut réussi !", "Saut de qualité !"]

astuces = ["Diminue la longueur de la perche", 
           "Diminue un peu la raideur ou la longueur de la perche", 
           "Augmente légèrement la raideur", 
           "Ton équilibre tu as trouvé !"]

# Affichage
fig, axis = plt.subplots(1, 1, num = 'Saut a la perche')
fig.set_size_inches(10,6)
axis.set_xlabel(r'$x$')
axis.set_ylabel(r'$z$')
axis.set_ylim([0, 7])
axis.set_xlim([-5, 4])
axis.set_aspect('equal')
axis.set_title('Trajectoire')
plt.axhline(y=WR, color='gold', linestyle='--')
plt.text(-4, WR, "WR 6.22 m", ha='left', va='bottom')
line = []

# valeurs initiales des paramètres
m_ini = 80
H_ini = 1.80
v0_ini = 10
Lp_ini = 3.5
EI0_min=int(min_EI(m_ini, Lp_ini))
EI0_max=int(max_EI(m_ini, Lp_ini, v0_ini, k))
EI0_step=int((EI0_max-EI0_min)/20)
EI0_ini=EI0_max - EI0_step

################### STREAMLIT 

st.title("Saut à la Perche")

############# On initie la mémoire cache #################
if 'EI_min' not in st.session_state:
    st.session_state.EI_min = EI0_min
    
if 'EI_max' not in st.session_state:
    st.session_state.EI_max = EI0_max

if 'EI_step' not in st.session_state:
    st.session_state.EI_step = EI0_step
if 'EI_ini' not in st.session_state:
    st.session_state.EI_ini = EI0_ini
    
if 'hint' not in st.session_state:
    st.session_state.hint = 'La patience est une force...'

#Fonction qui met à jour les paramètres du widget de raideur. Et permet de garder le curseur sur la même valeur (sauf si elle n'existe plus)
def set_update():
      if ('m_widget' in st.session_state) and ('Lp_widget' in st.session_state) and ('v0_widget' in st.session_state):
          st.session_state.EI_max=int(max_EI(st.session_state.m_widget, st.session_state.Lp_widget, st.session_state.v0_widget, k))
          st.session_state.EI_min=int(min_EI(st.session_state.m_widget, st.session_state.Lp_widget))
          st.session_state.EI_step=int((st.session_state.EI_max-st.session_state.EI_min)/20)
          if st.session_state.EI_ini<st.session_state.EI_min:
              st.session_state.EI_ini=st.session_state.EI_min
          elif st.session_state.EI_ini>st.session_state.EI_max:
              st.session_state.EI_ini=st.session_state.EI_max
    
with st.sidebar: 
    st.title('Paramètres')
    st.write('Le perchiste')
m_widget = st.sidebar.slider('Choisis ta masse',
                      min_value=50, max_value=100,value=m_ini,
                      step=5,key='m_widget',on_change=set_update())
H_widget = st.sidebar.slider('Choisis ta taille',
                      min_value=1.50, max_value=2.20,value=H_ini,
                      step=0.1,key='H_widget',on_change=set_update())
v0_widget = st.sidebar.slider('Choisis ta rapidité', 
                      min_value=5, max_value=10,value=v0_ini,
                      step=1,key='v0_widget',on_change=set_update())
with st.sidebar: 
    st.write('Son équipement')

Lp_widget = st.sidebar.slider('Choisis la longueur de ta perche',
                      min_value=3., max_value=5.,value=Lp_ini,
                      step=0.1,key='Lp_widget',on_change=set_update())
EI_widget = st.sidebar.slider('Choisis la raideur',
                      min_value= st.session_state.EI_min, max_value= st.session_state.EI_max, value= st.session_state.EI_ini,
                      step= st.session_state.EI_step,key='EI_widget')
         
def Affiche():
    m = st.session_state.m_widget
    H = st.session_state.H_widget
    v0 = st.session_state.v0_widget
    Lp = st.session_state.Lp_widget
    EI = st.session_state.EI_widget
    
    F0 = np.pi**2 * EI / (Lp**2)
    H0 = 0.6 * H
    K = F0 / (m*g)
    L = Lp / H0
    for i in range(len(line)):
        Axis.set_visible(line[i], False)
        
    # Trajectoire
    y_flag, sol, sol_flight = classifyJump(K,L,H0,v0)
    if y_flag in [2, 3]:
        X = np.hstack([sol.y[0], sol_flight.y[0]])
        Z = np.hstack([sol.y[1], sol_flight.y[1]])
    else:
        X = sol.y[0]
        Z = sol.y[1] 
    line.append(axis.plot(X*H0, Z*H0, color='dodgerblue')[0])
    st.header(message[y_flag])
    with st.empty():
        st.pyplot(fig,use_container_width=True)
        # if st.button('animation'):
        Xp = []
        Yp = []
        time = np.hstack([sol.t, sol_flight.t])
        count = 0
        for j in range(len(X)):
            if time[j] >= count * 0.05 or j==len(X)-1: # calcul de la déformée toutes les 50 ms --> 20 fps
                count += 1
                X_temp, Y_temp = perche_deformee(X[j], Z[j], Lp)
                Xp.append(X_temp)
                Yp.append(Y_temp)
    
        # animation
        perche, = axis.plot([], [], color='tab:red')
        def animate(i):
            perche.set_data(Xp[i], Yp[i])
            return perche,
        anim = animation.FuncAnimation(fig, animate, len(Xp), interval=50, blit=True, repeat=False)
        components.html(anim.to_jshtml(), height=700, width=1000)
    placeholder = st.empty()
    with tab3:
        st.write(astuces[y_flag])


tab1,tab2,tab3=st.tabs(['Tuto','Trajectoires','Astuces'])


# image = st.file_uploader("Diagramme", type="png")
# image = Image.open('./Diagramme.png')

with tab1:
    
    st.header("Un petit tuto s'impose")
    st.write("Tu trouveras sur la barre latérale différents paramètres associés à l'athlète et sa perche.")
    fm_widget = st.slider("C'est un exemple",
                      min_value=50, max_value=100,value=m_ini,
                      step=5)
    # fH_widget = st.slider('bla',
    #                   min_value=1.50, max_value=2.20,value=H_ini,
    #                   step=0.1)
    # fv0_widget = st.slider('bla', 
    #                   min_value=5, max_value=10,value=v0_ini)
    # fLp_widget = st.slider('bla',
    #                   min_value=3., max_value=5.,value=Lp_ini,
    #                   step=0.1)
    # EI_widget = st.slider('bla',
    #                   min_value= EI0_min, max_value=EI0_max, value=EI0_ini,
    #                   step= EI0_step)
    st.write("Ensuite, tu peux cliquer sur l'onglet Trajectoires: Tu peux apercevoir la trajectoire du perchiste. Si tu attends encore un peu, une vidéo apparaîtra et te permettra de voir la perche en action!")
    st.write(" En utilisant des perches tu peux décupler ta capacité de saut ! Aujourd’hui le record du monde de saut à la perche est à 6.22 m. Pour réaliser une telle prouesse il faut choisir une perche adaptée et avoir une technique irréprochable. Avant de sauter, une bonne course permet d’accumuler de l’énergie, que tu peux transmettre à la perche en la pliant. La perche te redonnera cette énergie pour te permettre de prendre de la hauteur. Avec une bonne technique tu peux même rajouter de l’énergie dans la perche pendant le saut ! Essayons déjà de trouver une perche qui te convient sans penser à la technique.")
    
    st.write("Si tu veux faire un bon saut, il faut bien déjà bien choisir ta perche. Si elle est trop raide ou trop longue tu risques d’être renvoyé vers la piste. Si elle est trop souple tu ne sauteras pas bien haut. Pour sauter haut c’est mieux d’avoir une perche longue. Mais attention ! Comme tu peux le voir sur ce graphique, plus la perche est longue et plus ce sera difficile de trouver la bonne raideur.")
    st.image("Diagramme.png")
    st.write("Pour t’aider tu ne pourras choisir la raideur qu’entre deux valeurs. La perche doit être assez souple pour pouvoir la plier avec l’énergie accumulée grâce à la course, et assez raide pour qu’elle se déplie pendant le saut.")
    
    st.header("Un bon perchiste sait trouver son équilibre... et toi?")
    

with tab2:
    Affiche()
    
    
