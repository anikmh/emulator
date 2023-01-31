#ifndef O2SCL_GPR_PYTHON__H
#define O2SCL_GPR_PYTHON__H

#include <iostream>
#include <string>
#include <cmath>

#include <boost/numeric/ublas/matrix.hpp>

#include <gsl/gsl_combination.h>

#include <o2scl/err_hnd.h>
#include <o2scl/vector.h>
#include <o2scl/vec_stats.h>
#include <o2scl/linear_solver.h>
#include <o2scl/columnify.h>
#include <o2scl/table.h>

#include <Python.h>

using namespace std;

namespace o2scl{

  template<class vec_t=ubvector, class vec2_t=ubvector>
  class emulator_p {

    protected:
      /// The Python Module name
      string pmod_name;
      /// The Python class name
      string pcls_name;
      /// The number of dimensions of the inputs
      size_t nd_in;
      /// The full parameter List
      vector<std::string> param_list;

      PyObject *emu_gpr;      // Use to import_module emu.py
      PyObject *m_gpr;        // Use to create python class object modgpr
      PyObject *i_gpr;        // instance of the modgpr class
      PyObject *emu_pred_gpr; // Call Python Predict function

    public:

      emulator_p() {
        emu_gpr = NULL;
        m_gpr = NULL;
        i_gpr = NULL;
        emu_pred_gpr = NULL;
      }

      void set_data(vector<string> files, string mod_name, string cls_name, size_t n_in,
                 vector<string> list) {
                
        pmod_name = mod_name;
        pcls_name =cls_name;
        nd_in=n_in;
        param_list=list;

        if (!Py_IsInitialized()) {
          PyRun_SimpleString("print 'inital error!' ");
          return;
        }
 
        emu_gpr = PyImport_ImportModule(pmod_name.c_str());
        m_gpr = PyObject_GetAttrString(emu_gpr, pcls_name.c_str());
        i_gpr = PyObject_CallObject(m_gpr, 0);

        PyObject *py_file_list = PyList_New(files.size());
        for (size_t i=0;i<files.size();i++) {
          PyList_SetItem(py_file_list, i, PyUnicode_FromString(files[i].c_str()));
        }
        
        PyObject *py_param_list = PyList_New(param_list.size());
        for (size_t i=0;i<param_list.size();i++) {
          PyList_SetItem(py_param_list, i, PyUnicode_FromString(param_list[i].c_str()));
        }

        PyObject *py_nd_in = PyLong_FromSize_t(nd_in);
        PyObject *read_data_python = PyObject_GetAttrString(i_gpr, "read_data");
        PyObject *info_gpr = PyTuple_Pack(4,py_file_list,
                    PyUnicode_FromString("markov_chain_0"),py_nd_in, py_param_list);

        
        PyObject_CallObject(read_data_python, info_gpr);
        
        PyObject *emu_train_gpr = PyObject_GetAttrString(i_gpr, "modTrain");
        PyObject_CallObject(emu_train_gpr, 0);
        
        emu_pred_gpr = PyObject_GetAttrString(i_gpr, "predict"); 
        Py_DECREF(py_nd_in); Py_DECREF(py_param_list); Py_DECREF(read_data_python); Py_DECREF(info_gpr);
        Py_DECREF(emu_train_gpr);  
        PyErr_Print();  
        return;
      }

      void upTrain(const std::vector<double> &p) {

        PyObject *emu_up_dat = PyList_New(p.size());
        for (size_t i=0;i<p.size();i++) {  
          PyList_SetItem(emu_up_dat, i, PyFloat_FromDouble(p[i]));
        }

        PyObject *emu_uptrain = PyObject_GetAttrString(i_gpr, "upTrain");
        PyObject_CallObject(emu_uptrain, PyTuple_Pack(1, emu_up_dat));
        Py_DECREF(emu_uptrain); Py_DECREF(emu_up_dat);
        PyErr_Print(); 
      }

      int eval_unc(const vec_t &p, vec2_t &dat, vec2_t &dat_unc) {  

        PyObject *tr_in_gpr = PyList_New(p.size());
        for (size_t i=0;i<p.size();i++) {  
          PyList_SetItem(tr_in_gpr, i, PyFloat_FromDouble(p[i]));
        }
        
        PyObject *result_gpr = PyObject_CallObject(emu_pred_gpr, PyTuple_Pack(1, tr_in_gpr));
        
        if (result_gpr == NULL) {
            std::cout << "no result" << std::endl;
            return 0;
        } else {
          size_t nd_out = param_list.size()-nd_in;
          // result and std_dev are stored in alternate order in the pyobject result_gpr
          for (size_t i=0; i < nd_out; i++) {
            dat[i] = PyFloat_AsDouble(PyList_GetItem(result_gpr, static_cast<int>(2*i)));
            dat_unc[i]= PyFloat_AsDouble(PyList_GetItem(result_gpr, static_cast<int>(2*i+1)));
            //std::cout << param_list[nd_in+i] << ": " << dat[i] << ", " << dat_unc[i] << std::endl;
          }
        }
 
        Py_DECREF(tr_in_gpr);  Py_DECREF(result_gpr);
        PyErr_Print();
        //Py_Finalize();
        return 0;     
      }
  };
}
#endif