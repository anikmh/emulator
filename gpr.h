#ifndef GPR_H
#define GPR_H

#include "./gpr_python.h"

using namespace std;

namespace o2scl {

  typedef boost::numeric::ublas::vector<double> ubvector;

  template<class vec2_t=ubvector, class vec_t=ubvector>
  class emulator_prs :
    public emulator_unc<vec2_t,vec2_t,vec_t> {

  protected:
    
    /// Index of the "log weight" in the MCMC data vector
    size_t ix;

  public:

    o2scl::emulator_p<vec2_t,vec_t> ep;
    
    /** \brief Create an emulator
     */
    emulator_prs() {
    }
    
    /** \brief Set the emulator

        Set the emulator using a table containing \c np parameters and
        \c n_out output quantities. The variable \c ix_log_wgt should
        be the index of the log_weight among all of the output
        variables, from 0 to <tt>n_out-1</tt>. The list, \c list,
        should include the column names of the parameters and then the
        output quantities (including the log weight column), in order.
     */
    void set(std::vector<std::string> files, string mod_name, string cls_name, size_t n_in, 
              size_t ix_log_wgt, std::vector<std::string> list) {
      //cmvtt.set(t,list);
      ix=ix_log_wgt;
      ep.set_data(files, mod_name, cls_name, n_in, list);
      return;
    }

    void update(const std::vector<double> &p) {
      ep.upTrain(p);
      return;
    }
    
    /** \brief Evaluate the emulator at the point \c p returning
        \c log_wgt and \c dat and their uncertainties
     */
    virtual int eval_unc(size_t n, const vec_t &p, double &log_wgt,
                 double &log_wgt_unc, vec2_t &dat, vec2_t &dat_unc) {
      
      ep.eval_unc(p,dat,dat_unc);
      log_wgt=dat[ix];
      log_wgt_unc=dat_unc[ix];
      return 0;
    }
    
  };

}

#endif