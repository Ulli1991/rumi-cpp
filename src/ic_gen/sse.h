#pragma once

struct SSEResult {
    int kw;           // Stellar type (0-15)
    double mass;      // Final mass (M_sun)
    double mass_core; // Core mass (M_sun)
};

class SSEContext {
public:
    explicit SSEContext(double Z);
    SSEResult evolve(double m_zams, double age_myr) const;

private:
    double Z_;
    double zpars_[20];
    double msp_[200];
    double gbp_[200];

    // McLuster default parameters
    static constexpr double neta_   = 0.5;
    static constexpr double bwind_  = 0.0;
    static constexpr double hewind_ = 1.0;
    static constexpr double mxns_   = 3.0;
    static constexpr int    nsflag_ = 1;
    static constexpr int    ifflag_ = 1;
    static constexpr int    wdflag_ = 1;
    static constexpr double pts1_   = 0.05;
    static constexpr double pts2_   = 0.01;
    static constexpr double pts3_   = 0.02;

    void init_coefficients();

    // Fitting formulae from zfuncs.f (all use msp_/gbp_ arrays)
    double lzamsf(double m) const;
    double rzamsf(double m) const;
    double tbgbf(double m) const;
    double thookf(double m) const;
    double ltmsf(double m) const;
    double lalphf(double m) const;
    double lbetaf(double m) const;
    double lnetaf(double m) const;
    double lhookf(double m, double mhook) const;
    double rtmsf(double m) const;
    double ralphf(double m) const;
    double rbetaf(double m) const;
    double rgammf(double m) const;
    double rhookf(double m, double mhook) const;
    double lbgbf(double m) const;
    double lbagbf(double m, double mhefl) const;
    double rgbf(double m, double lum) const;
    double ragbf(double m, double lum, double mhelf) const;
    double mctmsf(double m) const;
    double mcheif(double m, double mhefl, double mchefl) const;
    double mcagbf(double m) const;
    double mbagbf(double mc) const;
    double mcgbtf(double t, double A, const double GB[10],
                  double tinf1, double tinf2, double tx) const;
    double lgbtf(double t, double A, const double GB[10],
                 double tinf1, double tinf2, double tx) const;
    double mcgbf(double lum, const double GB[10], double lx) const;
    double lmcgbf(double mc, const double GB[10]) const;
    double lheif(double m, double mhefl) const;
    double lhef(double m) const;
    double rminf(double m) const;
    double thef(double m, double mc, double mhefl) const;
    double tblf(double m, double mhefl, double mfgb) const;
    double lzahbf(double m, double mc, double mhefl) const;
    double rzahbf(double m, double mc, double mhefl) const;
    double lzhef(double m) const;
    double rzhef(double m) const;
    double themsf(double m) const;
    double rhehgf(double m, double lum, double rx, double lx) const;
    double rhegbf(double lum) const;
    double lpertf(double m, double mew) const;
    double rpertf(double m, double mew, double r, double rc) const;

    // Main routines
    void star(int& kw, double mass, double mt, double& tm, double& tn,
              double tscls[20], double lums[10], double GB[10]) const;
    void hrdiag(double& mass, double& aj, double& mt, double& tm, double& tn,
                double tscls[20], double lums[10], double GB[10],
                double& r, double& lum, int& kw, double& mc, double& rc,
                double& menv, double& renv) const;
    double mlwind(int kw, double lum, double r, double mt, double mc,
                  double rl, double z) const;
    void deltat(int kw, double age, double tm, double tn,
                const double tscls[20], double& dt, double& dtr) const;
};
