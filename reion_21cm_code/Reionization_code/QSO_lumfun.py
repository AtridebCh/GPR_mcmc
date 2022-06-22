import numpy as np
import math
from scipy.integrate import quad

def lumfun_integrand(lb,z,logphistar,loglstar,gamma1,gamma2):
	x=lb
	psi=10.0**logphistar/(x**gamma1+x**gamma2)
	quasar_lumfun=10.0**loglstar*psi/(math.log(10.0)*lb)
	lumfun_integrand=quasar_lumfun*lb
	lstarby10=10.0**loglstar*1e-10
	lband=8.99833*(x*lstarby10)**-0.0115970+6.24800*(x*lstarby10)**-0.370587
	return lumfun_integrand/lband

def lumfun_integral(z):
	P0,P1,P2,P3,P4,P5,P6,P7,P8,P9=-4.8250643, 13.035753, 0.63150872, -11.763560, -14.249833, 0.41698725, -0.62298947, 2.1744386,  1.4599393, -0.79280099
	if z>12.0:
		return 0.0
	betamin=1.30
	xi=math.log10((1.0+z)/3.0)
	logphistar=P0
	loglstar=P1+P2*xi+P3*xi*xi+P4*xi*xi*xi
	gamma1=P5*10.0**(P6*xi)
	gamma2=2.0*P7/(10.0**(P8*xi)+10.0**(P9*xi))
	gamma2=max(betamin,gamma2)
	return quad(lumfun_integrand,0.0,np.inf,args=(z,logphistar,loglstar,gamma1,gamma2),epsabs=1.49e-03, epsrel=1.49e-03)[0]

