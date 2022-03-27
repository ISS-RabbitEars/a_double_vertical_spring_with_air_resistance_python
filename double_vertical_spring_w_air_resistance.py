import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(ic, ti, p):
	y1, v1, y2, v2 = ic
	m1, m2, k1, k2, yeq1, yeq2, gc, rho, cd, ar = p

	print(ti)

	return [v1, AY1.subs({M1:m1, K1:k1, K2:k2, YEQ1:yeq1, YEQ2:yeq2, g:gc, Y1:y1, Y2:y2, Y1dot:v1, RHO:rho, CD:cd, Ar:ar}),\
		v2, AY2.subs({M2:m2, K2:k2, YEQ2:yeq2, g:gc, Y1:y1, Y2:y2, Y2dot:v2, RHO:rho, CD:cd, Ar:ar})]


M1, M2, K1, K2, YEQ1, YEQ2, g, t = sp.symbols('M1 M2 K1 K2 YEQ1 YEQ2 g t')
RHO, CD, Ar = sp.symbols('RHO CD Ar')
Y1, Y2 = dynamicsymbols('Y1 Y2')

Y1dot = Y1.diff(t, 1)
Y2dot = Y2.diff(t, 1)

T = sp.Rational(1, 2) * (M1 * Y1dot**2 + M2 * Y2dot**2)
V = sp.Rational(1, 2) * (K1 * (Y1 - YEQ1)**2 + K2 * (Y2 - Y1 - YEQ2)**2) + g * (M1 * Y1 + M2 * Y2)

L = T - V

dLdY1 = L.diff(Y1, 1)
dLdY1dot = L.diff(Y1dot, 1)
ddtdLdY1dot = dLdY1dot.diff(t, 1)
dLdY2 = L.diff(Y2, 1)
dLdY2dot = L.diff(Y2dot, 1)
ddtdLdY2dot = dLdY2dot.diff(t, 1)

Fc = sp.Rational(1, 2) * RHO * CD * Ar
F1 = Fc * sp.sign(Y1dot) * Y1dot**2
F2 = Fc * sp.sign(Y2dot) * Y2dot**2

dL1 = ddtdLdY1dot - dLdY1 + F1
dL2 = ddtdLdY2dot - dLdY2 + F2

sol1 = sp.solve(dL1, Y1.diff(t, 2))
sol2 = sp.solve(dL2, Y2.diff(t, 2))

AY1 = sol1[0]
AY2 = sol2[0]

#---------------------------------------------------

gc = 9.8
m1, m2 = [2, 2]
k1, k2 = [50, 50] 
yeq1, yeq2 = [-2.5, -2.5]
y1o, y2o = [-2.5, -5]
v1o, v2o = [0, 0]
rho = 1.225
cd = 0.47
rad = 0.25
ar = np.pi * rad**2

p = m1, m2, k1, k2, yeq1, yeq2, gc, rho, cd, ar
ic = y1o, v1o, y2o, v2o

tf = 240 
nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

yv = odeint(integrate, ic, ta, args=(p,))

y1 = yv[:,0]
v1 = yv[:,1]
y2 = yv[:,2]
v2 = yv[:,3]

ke = np.asarray([T.subs({M1:m1, M2:m2, Y1dot:v1[i], Y2dot:v2[i]}) for i in range(nframes)])
pe = np.asarray([V.subs({M1:m1, M2:m2, K1:k1, K2:k2, YEQ1:yeq1, YEQ2:yeq2, g:gc, Y1:y1[i], Y2:y2[i]}) for i in range(nframes)])
E = ke + pe

fig, a=plt.subplots()

xline = 0
xmax = xline + 2 * rad
xmin = xline - 2 * rad
ymax = 2 * rad
ymin = min(y2) - 2 * rad
dy12 = np.asarray([y1[i] - y2[i] for i in range(nframes)])
nl1 = int(np.ceil((max(np.abs(y1))+rad)/(2*rad)))
nl2 = int(np.ceil(max(dy12)/(2*rad)))
xl1 = np.zeros((nl1,nframes))
yl1 = np.zeros((nl1,nframes))
xl2 = np.zeros((nl2,nframes))
yl2 = np.zeros((nl2,nframes))
for i in range(nframes):
	l1 = (np.abs(y1[i])/nl1)
	l2 = (y1[i]-y2[i]-2*rad)/nl2
	yl1[0][i] = y1[i] + rad + 0.5*l1
	yl2[0][i] = y2[i] + rad + 0.5*l2
	for j in range(1,nl1):
		yl1[j][i] = yl1[j-1][i] + l1
	for j in range(1,nl2):
		yl2[j][i] = yl2[j-1][i] + l2
	for j in range(nl1):
		xl1[j][i] = xline+((-1)**j)*(np.sqrt(rad**2 - (0.5*l1)**2))
	for j in range(nl2):
		xl2[j][i] = xline+((-1)**j)*(np.sqrt(rad**2 - (0.5*l2)**2))

def run(frame):
	plt.clf()
	plt.subplot(181)
	circle=plt.Circle((xline,y1[frame]),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	circle=plt.Circle((xline,y2[frame]),radius=rad,fc='xkcd:light purple')
	plt.gca().add_patch(circle)
	plt.plot([xline,xl1[0][frame]],[y1[frame]+rad,yl1[0][frame]],'xkcd:cerulean')
	plt.plot([xl1[nl1-1][frame],xline],[yl1[nl1-1][frame],rad],'xkcd:cerulean')
	for i in range(nl1-1):
		plt.plot([xl1[i][frame],xl1[i+1][frame]],[yl1[i][frame],yl1[i+1][frame]],'xkcd:cerulean')
	plt.plot([xline,xl2[0][frame]],[y2[frame]+rad,yl2[0][frame]],'xkcd:cerulean')
	plt.plot([xl2[nl2-1][frame],xline],[yl2[nl2-1][frame],y1[frame]-rad],'xkcd:cerulean')
	for i in range(nl2-1):
		plt.plot([xl2[i][frame],xl2[i+1][frame]],[yl2[i][frame],yl2[i+1][frame]],'xkcd:cerulean')
	plt.title("A Double Vertical Spring\nWith Air Resistance")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(1,8,(2,8))
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=0.5)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=0.5)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')
	ax.yaxis.set_label_position("right")
	ax.yaxis.tick_right()

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('double_vertical_spring_w_air_resistance.mp4', writer=writervideo)
plt.show()
