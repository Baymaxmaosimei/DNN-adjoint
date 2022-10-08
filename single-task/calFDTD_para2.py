##calculation on FDTD

import sys
import os
import inspect
import shutil
import numpy as np
import lumapi
# import matplotlib.pyplot as plt


class calFDTD(object):
    """ Performing Forward and adjoint simulations. Calculating the accurate transmission and gradient with respect to params,
        which requires three key pieces:
            1) a script for simulation,
            2) an object that define and collects the FOM
            3) an object that calculate and collects the gradients 
        
        Parameters:
        :param base_script:    callable, file name of the base simulation.
        :param wavelengths:    wavelength value (float) or range (class Wavelengths) with the spectral range for all simulations. 
    """
    def __init__(self, base_script, xyz, wavelengths, beta=1):
        self.base_script = base_script
        self.wavelengths = wavelengths
        self.wavelength_range = self.wavelengths.max() - self.wavelengths.min()
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.dz = self.z[1]-self.z[0]
        self.norm_p = 2
        self.eps_min = 1.447**2
        self.eps_max = 3.5**2
        self.iteration = 1
        self.beta = beta
        self.eta =0.5
        self.best_fom = 0
        self.bianry_loss = 0

        frame = inspect.stack()[1]
        calling_file_name = os.path.abspath(frame[0].f_code.co_filename)
        self.goto_new_opts_folder(calling_file_name, base_script)
        self.workingDir = os.getcwd()

    def initialize(self):
        # initialize the FDTD and load script
        self.fdtd = lumapi.FDTD(hide=0)
        self.fdtd.cd(self.workingDir)
        self.fdtd.load(self.base_script)

        self.fdtd.setglobalmonitor('use source limits', True)
        self.fdtd.setglobalmonitor('use wavelength spacing', True)
        self.fdtd.setglobalmonitor('frequency points', len(self.wavelengths))
        self.fdtd.setglobalsource('set wavelength', True)
        self.fdtd.setglobalsource('wavelength start', self.wavelengths.min())
        self.fdtd.setglobalsource('wavelength stop', self.wavelengths.max())


    def callable_fom_grad(self,params, targets):
        """ Function for the optimizers to retrieve the FOM
            :param params: optimization parameters
            :param returns: figure of merit 
        """
        """ Generates the new forward simulations, runs them and computes the figure of merit and forward fields. """

        self.num_para = np.size(params,0)
        T_fwd_vs_wavelength = []
        self.bianry_loss = []
        gradients = []

        ## making solvers
        for ii in range(self.num_para):
            self.make_sim(params[ii], name = 'forward', iter = ii)
            self.make_sim(params[ii], name = 'adjoint', iter = ii)

        # run all the solvers
        for ii in range(self.num_para-1):
            self.fdtd.addjob('adjoint_{}'.format(ii))
        self.fdtd.run()

        # record all the transmission and electric fields
        for jj in range(self.num_para):
            self.fdtd.load('forward_{}'.format(jj))
            self.fdtd.eval("forward_fields = getresult('opt_fields','E');")
            self.fdtd.eval( "mode_exp = getresult('fom_mode_exp', 'expansion for fom_mode_exp');"+
                            "T_forward = mode_exp.T_forward;"+
                            "source_power = sourcepower(mode_exp.f);"+
                            "phase_prefactors = mode_exp.a*sqrt(mode_exp.N)/4.0/source_power;")
            T_fwd_vs_wavelength.append(self.fdtd.getv("T_forward").flatten())

            self.fdtd.load('adjoint_{}'.format(jj))
            self.fdtd.eval("adjoint_fields = getresult('opt_fields','E');")
            self.fdtd.eval( "adjoint_source_power = sourcepower(adjoint_fields.f);"+
                            "scaling_factor = conj(phase_prefactors)*2*pi*adjoint_fields.f*1i/sqrt(adjoint_source_power);"
                            "V_cell = {};".format(self.dx*self.dy*self.dz) +
                            "dF_dEps = sum(sum(2.0 * V_cell * eps0 * forward_fields.E * adjoint_fields.E,5),3);" +
                            "num_wl_pts = length(forward_fields.lambda);" +
                            "for(wl_idx = [1:num_wl_pts]){" +
                            "    dF_dEps(:,:,wl_idx) = dF_dEps(:,:,wl_idx) * scaling_factor(wl_idx);" +
                            "}" + 
                            "dF_dEps = real(dF_dEps);")
            gradients_sig = self.fdtd.getv("dF_dEps")*(self.eps_max-self.eps_min)/2
            gradients.append(gradients_sig)

        # calculate the mean FOM
        T_fwd_vs_wavelength = np.array(T_fwd_vs_wavelength)
        T_fwd_error = T_fwd_vs_wavelength-targets
        T_fwd_error_integrand = np.power(np.abs(T_fwd_error),self.norm_p)/self.wavelength_range
        error_term = np.power(np.trapz(y=T_fwd_error_integrand, x=self.wavelengths), 1.0/self.norm_p)
        fom = 1-error_term

        # integral gradients with respect to parameters
        const_factor = np.power(np.trapz(y=T_fwd_error_integrand, x=self.wavelengths), 1.0/self.norm_p-1.0)
        integral_kernel = np.power(np.abs(T_fwd_error), self.norm_p-1)*np.sign(T_fwd_error)/self.wavelength_range
        d = np.diff(self.wavelengths)
        quad_weight = np.append(np.append(d[0], d[0:-1]+d[1:]),d[-1])/2
        T_fwd_partial_derivs=[]
        for kk in range(self.num_para):
            v = const_factor[kk] * integral_kernel[kk] * quad_weight
            T_fwd_partial_derivs.append(gradients[kk].dot(v))
        T_fwd_partial_derivs=np.array(T_fwd_partial_derivs)

        return fom, T_fwd_vs_wavelength, T_fwd_partial_derivs.real, self.bianry_loss

    def callable_jac(self,params):
        """ Function for the optimizer to extract the figure of merit gradient.
            :param returns: partial derivative of the figure of merit with respect to each optimization parameter.
        """
        gradients = []
        # get the electric fields and calculate the gradients
        for kk in range(self.num_para):
            self.fdtd.load('forward_{}'.format(kk))
            self.fdtd.eval("forward_fields = getresult('opt_fields','E');")
            self.fdtd.load('adjoint_{}'.format(kk))
            self.fdtd.eval("adjoint_fields = getresult('opt_fields','E');")
            self.fdtd.eval( "adjoint_source_power = sourcepower(adjoint_fields.f);"+
                        "scaling_factor = conj(phase_prefactors)*2*pi*adjoint_fields.f*1i/sqrt(adjoint_source_power);"
                        "V_cell = {};".format(self.dx*self.dy*self.dz) +
                        "dF_dEps = sum(sum(2.0 * V_cell * eps0 * forward_fields.E * adjoint_fields.E,5),3);" +
                        "num_wl_pts = length(forward_fields.lambda);" +
                        "for(wl_idx = [1:num_wl_pts]){" +
                        "    dF_dEps(:,:,wl_idx) = dF_dEps(:,:,wl_idx) * scaling_factor(wl_idx);" +
                        "}" + 
                        "dF_dEps = real(dF_dEps);")
            gradients_sig = self.fdtd.getv("dF_dEps")*(self.eps_max-self.eps_min)/2
            gradients.append(gradients_sig)

        # integral gradients with respect to parameters
        const_factor = np.power(np.trapz(y=self.T_fwd_error_integrand, x=self.wavelengths), 1.0/self.norm_p-1.0)
        integral_kernel = np.power(np.abs(self.T_fwd_error), self.norm_p-1)*np.sign(self.T_fwd_error)/self.wavelength_range
        d = np.diff(self.wavelengths)
        quad_weight = np.append(np.append(d[0], d[0:-1]+d[1:]),d[-1])/2
        T_fwd_partial_derivs=[]
        for kk in range(self.num_para):
            v = const_factor[kk] * integral_kernel[kk] * quad_weight
            T_fwd_partial_derivs.append(gradients[kk].dot(v))

        T_fwd_partial_derivs=np.array(T_fwd_partial_derivs)
        return T_fwd_partial_derivs.real


    def make_sim(self, params, name, iter):
        self.fdtd.switchtolayout()
        self.fdtd.cd(self.workingDir)
        if name == 'forward':
            self.update_geometry(params)
            self.fdtd.setnamed('source','enabled',True)
            self.fdtd.setnamed('fom_mode_src','enabled',False)
            self.fdtd.save('{}_{}'.format(name,iter))
            self.fdtd.addjob('{}_{}'.format(name,iter))
      
        elif name == 'adjoint':
            self.fdtd.setnamed('source','enabled',False)
            self.fdtd.setnamed('fom_mode_src','enabled',True)
            self.fdtd.save('{}_{}'.format(name,iter))
                 


    def update_geometry(self, params):
        rho = np.reshape(np.float64(params.data.numpy()), (len(self.x),len(self.y)))
        eps = self.eps_min+(rho+1)/2*(self.eps_max-self.eps_min)
        # eps = self.eps_min+(np.tanh(self.beta*self.eta)+np.tanh(self.beta*(rho-self.eta)))/(np.tanh(self.beta*self.eta)+np.tanh(self.beta*(1-self.eta)))*(self.eps_max-self.eps_min)
        # eps = self.eps_min+(np.tanh(self.beta*self.eta)+np.tanh(self.beta*rho/2))/(np.tanh(self.beta*self.eta)+np.tanh(self.beta*(1-self.eta)))*(self.eps_max-self.eps_min)
        self.bianry_loss.append(np.mean((eps-self.eps_min)*(self.eps_max-eps)))
        # print('binary_loss: %.4f' %(bianry_loss))
        full_eps = np.broadcast_to(eps[:,:,None],(len(self.x),len(self.y),len(self.z)))
        self.fdtd.putv('x_geo',self.x)
        self.fdtd.putv('y_geo',self.y)
        self.fdtd.putv('z_geo',self.z)
        self.fdtd.putv('eps_geo',full_eps)
        script=('select("import");'
                'delete;'
                'addimport;'
                'importnk2(sqrt(eps_geo), x_geo, y_geo, z_geo);')
        self.fdtd.eval(script)


    @staticmethod
    def goto_new_opts_folder(calling_file_name, base_script):
        ''' Creates a new folder in the current working directory named opt_xx to store the project files of the
            various simulations run during the optimization. Backup copiesof the calling and base scripts are 
            placed in the new folder.'''

        calling_file_path = os.path.dirname(calling_file_name) if os.path.isfile(calling_file_name) else os.path.dirname(os.getcwd())
        calling_file_path_split = os.path.split(calling_file_path)
        if calling_file_path_split[1].startswith('opts_'):
            calling_file_path = calling_file_path_split[0]
        calling_file_path_entries = os.listdir(calling_file_path)
        opts_dir_numbers = [int(entry.split('_')[-1]) for entry in calling_file_path_entries if entry.startswith('opts_')]
        opts_dir_numbers.append(-1)
        new_opts_dir = os.path.join(calling_file_path, 'opts_{}'.format(max(opts_dir_numbers) + 1))
        os.mkdir(new_opts_dir)
        os.chdir(new_opts_dir)
        if os.path.isfile(calling_file_name):
            shutil.copy(calling_file_name, new_opts_dir)
        if hasattr(base_script, 'script_str'):
            with open('script_file.lsf','a') as file:
                file.write(base_script.script_str.replace(';',';\n'))




    
    
