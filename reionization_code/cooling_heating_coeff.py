#Heating and Cooling coefficient for different species
# Trivial functions hence not in class

from constants import kboltz, T_HI, T_HeI, T_HeII

def R_HII_e_A(T):
	return 0.0


def RC_HII_A(T):
	return 0.0

def R_HeII_e_A(T):
	return 0.0

def RC_HeII_A(T):
	return 0.0


def R_HeIII_e_A(T):
	return 0.0


	
def RC_HeIII_A(T):
	return 0.0
	
	

def R_HII_e_B(T):
	lambda_HI=2.0*T_HI/T
	return 2.753e-14*lambda_HI**1.5/(1.0+(lambda_HI/2.740)**0.407)**2.242


def RC_HII_B(T):
	lambda_HI=2.0*T_HI/T
	return 3.435e-30*T*lambda_HI**1.97/(1.0+(lambda_HI/2.250)**0.376)**3.72



def R_HeII_e_B(T):
	lambda_HeI=2.0*T_HeI/T
	return 1.26e-14*lambda_HeI**0.75


def RC_HeII_B(T):
	return kboltz*T*R_HeII_e_B(T)


def R_HeIII_e_B(T):
	lambda_HeII=2.0*T_HeII/T
	return 2.0*2.753e-14*lambda_HeII**1.5/(1.0+(lambda_HeII/2.74)**0.407)**2.242

def RC_HeIII_B(T):
	lambda_HeII=2.0*T_HeII/T
	return 8.0*3.435e-30*T*lambda_HeII**1.97/(1.0+(lambda_HeII/2.25)**0.376)**3.72
