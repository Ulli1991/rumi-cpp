#include "sse.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

// =========================================================================
// zdata.h - Initialization data for fitting formulae
// Fortran 1-based arrays stored as 0-based C++ arrays
// =========================================================================

// data for Lzams(0..34) and Rzams(35..75)
static const double xz[76] = {
    3.970417e-01, -3.2913574e-01, 3.4776688e-01, 3.7470851e-01, 9.011915e-02,
    8.527626e+00, -2.441225973e+01, 5.643597107e+01, 3.706152575e+01, 5.4562406e+00,
    2.5546e-04, -1.23461e-03, -2.3246e-04, 4.5519e-04, 1.6176e-04,
    5.432889e+00, -8.62157806e+00, 1.344202049e+01, 1.451584135e+01, 3.39793084e+00,
    5.563579e+00, -1.032345224e+01, 1.944322980e+01, 1.897361347e+01, 4.16903097e+00,
    7.8866060e-01, -2.90870942e+00, 6.54713531e+00, 4.05606657e+00, 5.3287322e-01,
    5.86685e-03, -1.704237e-02, 3.872348e-02, 2.570041e-02, 3.83376e-03,
    1.715359e+00, 6.2246212e-01, -9.2557761e-01, -1.16996966e+00, -3.0631491e-01,
    6.597788e+00, -4.2450044e-01, -1.213339427e+01, -1.073509484e+01, -2.51487077e+00,
    1.008855000e+01, -7.11727086e+00, -3.167119479e+01, -2.424848322e+01, -5.33608972e+00,
    1.012495e+00, 3.2699690e-01, -9.23418e-03, -3.876858e-02, -4.12750e-03,
    7.490166e-02, 2.410413e-02, 7.233664e-02, 3.040467e-02, 1.97741e-03,
    1.077422e-02,
    3.082234e+00, 9.447205e-01, -2.15200882e+00, -2.49219496e+00, -6.3848738e-01,
    1.784778e+01, -7.4534569e+00, -4.896066856e+01, -4.005386135e+01, -9.09331816e+00,
    2.2582e-04, -1.86899e-03, 3.88783e-03, 1.42402e-03, -7.671e-05
};

// data for Tbgb(0..16) and Thook(17..30)
static const double xt[31] = {
    1.593890e+03, 2.053038e+03, 1.231226e+03, 2.327785e+02,
    2.706708e+03, 1.483131e+03, 5.772723e+02, 7.411230e+01,
    1.466143e+02, -1.048442e+02, -6.795374e+01, -1.391127e+01,
    4.141960e-02, 4.564888e-02, 2.958542e-02, 5.571483e-03,
    3.426349e-01,
    1.949814e+01, 1.758178e+00, -6.008212e+00, -4.470533e+00,
    4.903830e+00, 5.212154e-02, 3.166411e-02, -2.750074e-03, -2.271549e-03,
    1.312179e+00, -3.294936e-01, 9.231860e-02, 2.610989e-02, 8.073972e-01
};

// data for Ltms, Lalpha, Lbeta, Lhook
static const double xl[72] = {
    1.031538e+00, -2.434480e-01, 7.732821e+00, 6.460705e+00, 1.374484e+00,
    1.043715e+00, -1.577474e+00, -5.168234e+00, -5.596506e+00, -1.299394e+00,
    7.859573e+02, -8.542048e+00, -2.642511e+01, -9.585707e+00,
    3.858911e+03, 2.459681e+03, -7.630093e+01, -3.486057e+02, -4.861703e+01,
    2.888720e+02, 2.952979e+02, 1.850341e+02, 3.797254e+01,
    7.196580e+00, 5.613746e-01, 3.805871e-01, 8.398728e-02,
    2.321400e-01, 1.828075e-03, -2.232007e-02, -3.378734e-03,
    1.163659e-02, 3.427682e-03, 1.421393e-03, -3.710666e-03,
    1.048020e-02, -1.231921e-02, -1.686860e-02, -4.234354e-03,
    1.555590e+00, -3.223927e-01, -5.197429e-01, -1.066441e-01,
    3.855707e-01, -6.104166e-01, 5.676742e+00, 1.060894e+01, 5.284014e+00,
    3.579064e-01, -6.442936e-01, 5.494644e+00, 1.054952e+01, 5.280991e+00,
    9.587587e-01, 8.777464e-01, 2.017321e-01,
    1.910302e-01, 1.158624e-01, 3.348990e-02, 2.599706e-03,
    3.931056e-01, 7.277637e-02, -1.366593e-01, -4.508946e-02,
    3.267776e-01, 1.204424e-01, 9.988332e-02, 2.455361e-02,
    5.990212e-01, 5.570264e-02, 6.207626e-02, 1.777283e-02
};

// data for Rtms, Ralpha, Rbeta, Rgamma, Rhook
static const double xr[119] = {
    2.187715e-01, -2.154437e+00, -3.768678e+00, -1.975518e+00, -3.021475e-01,
    1.466440e+00, 1.839725e+00, 6.442199e+00, 4.023635e+00, 6.957529e-01,
    2.652091e+01, 8.178458e+01, 1.156058e+02, 7.633811e+01, 1.950698e+01,
    1.472103e+00, -2.947609e+00, -3.312828e+00, -9.945065e-01,
    3.071048e+00, -5.679941e+00, -9.745523e+00, -3.594543e+00,
    -8.672073e-02,
    2.617890e+00, 1.019135e+00, -3.292551e-02, -7.445123e-02,
    1.075567e-02, 1.773287e-02, 9.610479e-03, 1.732469e-03,
    1.476246e+00, 1.899331e+00, 1.195010e+00, 3.035051e-01,
    5.502535e+00, -6.601663e-02, 9.968707e-02, 3.599801e-02,
    4.907546e-01, -1.683928e-01, -3.108742e-01, -7.202918e-02,
    4.537070e+00, -4.465455e+00, -1.612690e+00, -1.623246e+00,
    1.796220e+00, 2.814020e-01, 1.423325e+00, 3.421036e-01,
    2.256216e+00, 3.773400e-01, 1.537867e+00, 4.396373e-01,
    1.564231e-03, 1.653042e-03, -4.439786e-03, -4.951011e-03, -1.216530e-03,
    5.210157e+00, -4.143695e+00, -2.120870e+00,
    1.071489e+00, -1.164852e-01, -8.623831e-02, -1.582349e-02,
    7.108492e-01, 7.935927e-01, 3.926983e-01, 3.622146e-02,
    3.478514e+00, -2.585474e-02, -1.512955e-02, -2.833691e-03,
    3.969331e-03, 4.539076e-03, 1.720906e-03, 1.897857e-04,
    9.132108e-01, -1.653695e-01, 3.636784e-02,
    1.192334e-02, 1.083057e-02,
    1.230969e+00, 1.551656e+00, -1.668868e-01,
    5.818123e-01, -1.105027e+01, -1.668070e+01,
    7.615495e-01, 1.068243e-01, -2.011333e-01, -9.371415e-02,
    -1.015564e-01, -2.161264e-01, -5.182516e-02,
    -3.868776e-01, -5.457078e-01, -1.463472e-01,
    9.409838e+00, 1.522928e+00,
    7.330122e-01, 5.192827e-01, 2.316416e-01, 8.346941e-03,
    1.172768e+00, -1.209262e-01, -1.193023e-01, -2.859837e-02,
    3.982622e-01, -2.296279e-01, -2.262539e-01, -5.219837e-02,
    3.571038e+00, -2.223625e-02, -2.611794e-02, -6.359648e-03
};

// data for Lbgb, Lbagb, Rgb, Ragb, Mchei, Mcbagb
static const double xg[112] = {
    9.511033e+01, 6.819618e+01, -1.045625e+01, -1.474939e+01,
    3.113458e+01, 1.012033e+01, -4.650511e+00, -2.463185e+00,
    1.413057e+00, 4.578814e-01, -6.850581e-02, -5.588658e-02,
    3.910862e+01, 5.196646e+01, 2.264970e+01, 2.873680e+00,
    4.597479e+00, -2.855179e-01, 2.709724e-01,
    6.682518e+00, 2.827718e-01, -7.294429e-02,
    4.637345e+00, 9.301992e+00,
    1.626062e+02, -1.168838e+01, -5.498343e+00, 3.336833e-01,
    -1.458043e-01, -2.011751e-02,
    7.425137e+01, 1.790236e+01, 3.033910e+01, 1.018259e+01,
    9.268325e+02, -9.739859e+01, -7.702152e+01, -3.158268e+01,
    1.127018e+01, 1.622158e+00, -1.443664e+00, -9.474699e-01,
    2.474401e+00, 3.892972e-01,
    9.960283e-01, 8.164393e-01, 2.383830e+00, 2.223436e+00, 8.638115e-01,
    1.231572e-01, 2.561062e-01, 7.072646e-02, -5.444596e-02, -5.798167e-02, -1.349129e-02,
    1.157338e+00, 1.467883e+00, 4.299661e+00, 3.130500e+00, 6.992080e-01,
    1.640687e-02, 4.022765e-01, 3.050010e-01, 9.962137e-01, 7.914079e-01, 1.728098e-01,
    1.125124e+00, 1.306486e+00, 3.622359e+00, 2.601976e+00, 3.031270e-01,
    -1.343798e-01, 3.349489e-01, 4.531269e-03, 1.131793e-01, 2.300156e-01, 7.632745e-02,
    1.467794e+00, 2.798142e+00, 9.455580e+00, 8.963904e+00, 3.339719e+00,
    4.426929e-01, 4.658512e-01, 2.597451e-01, 9.048179e-01, 7.394505e-01, 1.607092e-01,
    1.110866e+00, 9.623856e-01, 2.735487e+00, 2.445602e+00, 8.826352e-01,
    1.140142e-01, -1.584333e-01, -1.728865e-01, -4.461431e-01, -3.925259e-01, -1.276203e-01, -1.308728e-02,
    9.796164e-02, 1.350554e+00,
    1.445216e-01, -6.180219e-02, 3.093878e-02, 1.567090e-02,
    1.304129e+00, 1.395919e-01, 4.142455e-03, -9.732503e-03,
    5.114149e-01, -1.160850e-02
};

// data for Lhei, Lhe, Rmin, The, Tbl, Lzahb, Rzahb
static const double xh[99] = {
    2.751631e+03, 3.557098e+02,
    -3.820831e-02, 5.872664e-02, 1.5e+01,
    1.071738e+02, -8.970339e+01, -3.949739e+01,
    7.348793e+02, -1.531020e+02, -3.793700e+01,
    9.219293e+00, -2.005865e+00, -5.561309e-01,
    2.917412e+00, 1.575290e+00, 5.751814e-01,
    6.371760e-01, 3.880523e-01,
    4.916389e+00, 2.862149e+00, 7.844850e-01,
    3.629118e+00, -9.112722e-01, 1.042291e+00,
    1.609901e+01, 7.391573e+00, 2.277010e+01, 8.334227e+00,
    1.747500e-01, 6.271202e-02, -2.324229e-02, -1.844559e-02,
    2.752869e+00, 2.729201e-02, 4.996927e-01, 2.496551e-01,
    -9.138012e-02, -3.671407e-01,
    3.518506e+00, 1.112440e+00, -4.556216e-01, -2.179426e-01,
    1.314955e+02, 2.009258e+01, -5.143082e-01, -1.379140e+00,
    1.823973e+01, -3.074559e+00, -4.307878e+00,
    1.5e+01,
    2.327037e+00, 2.403445e+00, 1.208407e+00, 2.087263e-01,
    1.079113e-01, 1.762409e-02, 1.096601e-02, 3.058818e-03,
    2.327409e+00, 6.901582e-01, -2.158431e-01, -1.084117e-01,
    1.997378e+00, -8.126205e-01,
    2.214315e+00, -1.975747e+00, 4.805428e-01,
    2.471620e+00, -5.401682e+00, 3.247361e+00,
    5.072525e+00, 1.146189e+01, 6.961724e+00, 1.316965e+00,
    5.139740e+00, 1.127733e+00, 2.344416e-01, -3.793726e-01,
    5.496045e+01, -1.289968e+01, 6.385758e+00, 1.832694e+00,
    -5.766608e-02, 5.696128e-02,
    1.211104e+02, 1.647903e+00,
    2.063983e+00, 7.363827e-01, 2.654323e-01, -6.140719e-02,
    2.214088e+02, 2.187113e+02, 1.170177e+01, -2.635340e+01,
    2.003160e+00, 9.388871e-01, 9.656450e-01, 2.362266e-01
};

// =========================================================================
// Notation: a[i] = msp_[i] (0-based), b[i] = gbp_[i] (0-based)
// Fortran msp(N) -> msp_[N-1], gbp(N) -> gbp_[N-1]
// Fortran xz(N)  -> xz[N-1],   xt(N) -> xt[N-1], etc.
// =========================================================================

// =========================================================================
// Constructor
// =========================================================================
SSEContext::SSEContext(double Z, const SSEConfig& cfg) : Z_(Z), cfg_(cfg) {
    std::fill(msp_, msp_ + 200, 0.0);
    std::fill(gbp_, gbp_ + 200, 0.0);
    std::fill(zpars_, zpars_ + 20, 0.0);
    init_coefficients();
}

// =========================================================================
// make_sse_config — parse string config values into SSEConfig
// =========================================================================
SSEConfig make_sse_config(const std::string& wind_str,
                          const std::string& remnant_str,
                          const std::string& ppisn_str,
                          const std::string& ecsn_str,
                          double mxns, double flbv) {
    SSEConfig c;
    c.mxns = mxns;
    c.flbv = flbv;

    if      (wind_str == "HURLEY")         c.wind = WindModel::HURLEY;
    else if (wind_str == "HURLEY_LBV")     c.wind = WindModel::HURLEY_LBV;
    else if (wind_str == "VINK")           c.wind = WindModel::VINK;
    else if (wind_str == "VINK_NO_BISTAB") c.wind = WindModel::VINK_NO_BISTAB;
    else throw std::runtime_error("Unknown SSE_WIND_MODEL: " + wind_str);

    if      (remnant_str == "HURLEY")        c.remnant = RemnantModel::HURLEY;
    else if (remnant_str == "BELCZYNSKI02")  c.remnant = RemnantModel::BELCZYNSKI02;
    else if (remnant_str == "BELCZYNSKI08")  c.remnant = RemnantModel::BELCZYNSKI08;
    else if (remnant_str == "FRYER_RAPID")   c.remnant = RemnantModel::FRYER_RAPID;
    else if (remnant_str == "FRYER_DELAYED") c.remnant = RemnantModel::FRYER_DELAYED;
    else if (remnant_str == "ELDRIDGE_TOUT") c.remnant = RemnantModel::ELDRIDGE_TOUT;
    else throw std::runtime_error("Unknown SSE_REMNANT_MODEL: " + remnant_str);

    if      (ppisn_str == "NONE")     c.ppisn = PPISNModel::NONE;
    else if (ppisn_str == "STRONG")   c.ppisn = PPISNModel::STRONG;
    else if (ppisn_str == "MODERATE") c.ppisn = PPISNModel::MODERATE;
    else if (ppisn_str == "WEAK")     c.ppisn = PPISNModel::WEAK;
    else throw std::runtime_error("Unknown SSE_PPISN_MODEL: " + ppisn_str);

    if      (ecsn_str == "OFF") c.ecsn = ECSNModel::OFF;
    else if (ecsn_str == "ON")  c.ecsn = ECSNModel::ON;
    else throw std::runtime_error("Unknown SSE_ECSN_MODEL: " + ecsn_str);

    return c;
}

// =========================================================================
// init_coefficients - port of zcnsts.f
// =========================================================================
void SSEContext::init_coefficients() {
    double z = Z_;
    double lzs = std::log10(z / 0.02);
    double dlzs = 1.0 / (z * std::log(10.0));
    double lz = std::log10(z);
    double lzd = lzs + 1.0;

    // c array used for zpars(7)
    double c[5] = {3.040581e-01, 8.049509e-02, 8.967485e-02,
                   8.780198e-02, 2.219170e-02};

    zpars_[0] = 1.0185 + lzs * (0.16015 + lzs * 0.0892);
    zpars_[1] = 1.995 + lzs * (0.25 + lzs * 0.087);
    zpars_[2] = 16.5 * std::pow(z, 0.06) / (1.0 + std::pow(1.0e-04 / z, 1.27));
    zpars_[3] = std::max(6.11044 + 1.02167 * lzs, 5.0);
    zpars_[4] = zpars_[3] + 1.8;
    zpars_[5] = 5.37 + lzs * 0.135;
    zpars_[6] = c[0] + lzs * (c[1] + lzs * (c[2] + lzs * (c[3] + lzs * c[4])));
    zpars_[7] = std::max(0.95, std::max(0.95 - (10.0 / 3.0) * (z - 0.01),
                std::min(0.99, 0.98 - (100.0 / 7.0) * (z - 0.001))));

    // Lzams: msp(1..7) -> msp_[0..6]
    msp_[0] = xz[0]+lzs*(xz[1]+lzs*(xz[2]+lzs*(xz[3]+lzs*xz[4])));
    msp_[1] = xz[5]+lzs*(xz[6]+lzs*(xz[7]+lzs*(xz[8]+lzs*xz[9])));
    msp_[2] = xz[10]+lzs*(xz[11]+lzs*(xz[12]+lzs*(xz[13]+lzs*xz[14])));
    msp_[3] = xz[15]+lzs*(xz[16]+lzs*(xz[17]+lzs*(xz[18]+lzs*xz[19])));
    msp_[4] = xz[20]+lzs*(xz[21]+lzs*(xz[22]+lzs*(xz[23]+lzs*xz[24])));
    msp_[5] = xz[25]+lzs*(xz[26]+lzs*(xz[27]+lzs*(xz[28]+lzs*xz[29])));
    msp_[6] = xz[30]+lzs*(xz[31]+lzs*(xz[32]+lzs*(xz[33]+lzs*xz[34])));

    // Rzams: msp(8..16) -> msp_[7..15]
    msp_[7]  = xz[35]+lzs*(xz[36]+lzs*(xz[37]+lzs*(xz[38]+lzs*xz[39])));
    msp_[8]  = xz[40]+lzs*(xz[41]+lzs*(xz[42]+lzs*(xz[43]+lzs*xz[44])));
    msp_[9]  = xz[45]+lzs*(xz[46]+lzs*(xz[47]+lzs*(xz[48]+lzs*xz[49])));
    msp_[10] = xz[50]+lzs*(xz[51]+lzs*(xz[52]+lzs*(xz[53]+lzs*xz[54])));
    msp_[11] = xz[55]+lzs*(xz[56]+lzs*(xz[57]+lzs*(xz[58]+lzs*xz[59])));
    msp_[12] = xz[60];
    msp_[13] = xz[61]+lzs*(xz[62]+lzs*(xz[63]+lzs*(xz[64]+lzs*xz[65])));
    msp_[14] = xz[66]+lzs*(xz[67]+lzs*(xz[68]+lzs*(xz[69]+lzs*xz[70])));
    msp_[15] = xz[71]+lzs*(xz[72]+lzs*(xz[73]+lzs*(xz[74]+lzs*xz[75])));

    // Tbgb: msp(17..21) -> msp_[16..20]
    msp_[16] = xt[0]+lzs*(xt[1]+lzs*(xt[2]+lzs*xt[3]));
    msp_[17] = xt[4]+lzs*(xt[5]+lzs*(xt[6]+lzs*xt[7]));
    msp_[18] = xt[8]+lzs*(xt[9]+lzs*(xt[10]+lzs*xt[11]));
    msp_[19] = xt[12]+lzs*(xt[13]+lzs*(xt[14]+lzs*xt[15]));
    msp_[20] = xt[16];

    // dTbgb/dz: msp(117..120) -> msp_[116..119]
    msp_[116] = dlzs*(xt[1]+lzs*(2.0*xt[2]+3.0*lzs*xt[3]));
    msp_[117] = dlzs*(xt[5]+lzs*(2.0*xt[6]+3.0*lzs*xt[7]));
    msp_[118] = dlzs*(xt[9]+lzs*(2.0*xt[10]+3.0*lzs*xt[11]));
    msp_[119] = dlzs*(xt[13]+lzs*(2.0*xt[14]+3.0*lzs*xt[15]));

    // Thook: msp(22..26) -> msp_[21..25]
    msp_[21] = xt[17]+lzs*(xt[18]+lzs*(xt[19]+lzs*xt[20]));
    msp_[22] = xt[21];
    msp_[23] = xt[22]+lzs*(xt[23]+lzs*(xt[24]+lzs*xt[25]));
    msp_[24] = xt[26]+lzs*(xt[27]+lzs*(xt[28]+lzs*xt[29]));
    msp_[25] = xt[30];

    // Ltms: msp(27..32) -> msp_[26..31]
    msp_[26] = xl[0]+lzs*(xl[1]+lzs*(xl[2]+lzs*(xl[3]+lzs*xl[4])));
    msp_[27] = xl[5]+lzs*(xl[6]+lzs*(xl[7]+lzs*(xl[8]+lzs*xl[9])));
    msp_[28] = xl[10]+lzs*(xl[11]+lzs*(xl[12]+lzs*xl[13]));
    msp_[29] = xl[14]+lzs*(xl[15]+lzs*(xl[16]+lzs*(xl[17]+lzs*xl[18])));
    msp_[26] = msp_[26]*msp_[29]; // msp(27) = msp(27)*msp(30)
    msp_[27] = msp_[27]*msp_[29]; // msp(28) = msp(28)*msp(30)
    msp_[30] = xl[19]+lzs*(xl[20]+lzs*(xl[21]+lzs*xl[22]));
    msp_[31] = xl[23]+lzs*(xl[24]+lzs*(xl[25]+lzs*xl[26]));

    // Lalpha: msp(33..42) -> msp_[32..41]
    double m2 = 2.0;
    msp_[32] = xl[27]+lzs*(xl[28]+lzs*(xl[29]+lzs*xl[30]));
    msp_[33] = xl[31]+lzs*(xl[32]+lzs*(xl[33]+lzs*xl[34]));
    msp_[34] = xl[35]+lzs*(xl[36]+lzs*(xl[37]+lzs*xl[38]));
    msp_[35] = xl[39]+lzs*(xl[40]+lzs*(xl[41]+lzs*xl[42]));
    msp_[36] = std::max(0.9, 1.1064 + lzs * (0.415 + 0.18 * lzs));
    msp_[37] = std::max(1.0, 1.19 + lzs * (0.377 + 0.176 * lzs));
    if (z > 0.01) {
        msp_[36] = std::min(msp_[36], 1.0);
        msp_[37] = std::min(msp_[37], 1.1);
    }
    msp_[38] = std::max(0.145, 0.0977 - lzs * (0.231 + 0.0753 * lzs));
    msp_[39] = std::min(0.24 + lzs * (0.18 + 0.595 * lzs), 0.306 + 0.053 * lzs);
    msp_[40] = std::min(0.33 + lzs * (0.132 + 0.218 * lzs), 0.3625 + 0.062 * lzs);
    msp_[41] = (msp_[32] + msp_[33] * std::pow(m2, msp_[35])) /
               (std::pow(m2, 0.4) + msp_[34] * std::pow(m2, 1.9));

    // Lbeta: msp(43..46) -> msp_[42..45]
    msp_[42] = xl[43]+lzs*(xl[44]+lzs*(xl[45]+lzs*(xl[46]+lzs*xl[47])));
    msp_[43] = xl[48]+lzs*(xl[49]+lzs*(xl[50]+lzs*(xl[51]+lzs*xl[52])));
    msp_[44] = xl[53]+lzs*(xl[54]+lzs*xl[55]);
    msp_[45] = std::min(1.4, 1.5135 + 0.3769 * lzs);
    msp_[45] = std::max(0.6355 - 0.4192 * lzs, std::max(1.25, msp_[45]));

    // Lhook: msp(47..51) -> msp_[46..50]
    msp_[46] = xl[56]+lzs*(xl[57]+lzs*(xl[58]+lzs*xl[59]));
    msp_[47] = xl[60]+lzs*(xl[61]+lzs*(xl[62]+lzs*xl[63]));
    msp_[48] = xl[64]+lzs*(xl[65]+lzs*(xl[66]+lzs*xl[67]));
    msp_[49] = xl[68]+lzs*(xl[69]+lzs*(xl[70]+lzs*xl[71]));
    msp_[50] = std::min(1.4, 1.5135 + 0.3769 * lzs);
    msp_[50] = std::max(0.6355 - 0.4192 * lzs, std::max(1.25, msp_[50]));

    // Rtms: msp(52..64) -> msp_[51..63]
    msp_[51] = xr[0]+lzs*(xr[1]+lzs*(xr[2]+lzs*(xr[3]+lzs*xr[4])));
    msp_[52] = xr[5]+lzs*(xr[6]+lzs*(xr[7]+lzs*(xr[8]+lzs*xr[9])));
    msp_[53] = xr[10]+lzs*(xr[11]+lzs*(xr[12]+lzs*(xr[13]+lzs*xr[14])));
    msp_[54] = xr[15]+lzs*(xr[16]+lzs*(xr[17]+lzs*xr[18]));
    msp_[55] = xr[19]+lzs*(xr[20]+lzs*(xr[21]+lzs*xr[22]));
    msp_[51] = msp_[51]*msp_[53]; // msp(52) = msp(52)*msp(54)
    msp_[52] = msp_[52]*msp_[53]; // msp(53) = msp(53)*msp(54)
    msp_[56] = xr[23];
    msp_[57] = xr[24]+lzs*(xr[25]+lzs*(xr[26]+lzs*xr[27]));
    msp_[58] = xr[28]+lzs*(xr[29]+lzs*(xr[30]+lzs*xr[31]));
    msp_[59] = xr[32]+lzs*(xr[33]+lzs*(xr[34]+lzs*xr[35]));
    msp_[60] = xr[36]+lzs*(xr[37]+lzs*(xr[38]+lzs*xr[39]));

    msp_[61] = std::max(0.097 - 0.1072 * (lz + 3.0),
               std::max(0.097, std::min(0.1461, 0.1461 + 0.1237 * (lz + 2.0))));
    msp_[61] = std::pow(10.0, msp_[61]);
    m2 = msp_[61] + 0.1;
    msp_[62] = (msp_[51] + msp_[52] * std::pow(msp_[61], msp_[54])) /
               (msp_[53] + std::pow(msp_[61], msp_[55]));
    msp_[63] = (msp_[56] * std::pow(m2, 3) + msp_[57] * std::pow(m2, msp_[60]) +
                msp_[58] * std::pow(m2, msp_[60] + 1.5)) / (msp_[59] + std::pow(m2, 5));

    // Ralpha: msp(65..76) -> msp_[64..75]
    msp_[64] = xr[40]+lzs*(xr[41]+lzs*(xr[42]+lzs*xr[43]));
    msp_[65] = xr[44]+lzs*(xr[45]+lzs*(xr[46]+lzs*xr[47]));
    msp_[66] = xr[48]+lzs*(xr[49]+lzs*(xr[50]+lzs*xr[51]));
    msp_[67] = xr[52]+lzs*(xr[53]+lzs*(xr[54]+lzs*xr[55]));
    msp_[68] = xr[56]+lzs*(xr[57]+lzs*(xr[58]+lzs*(xr[59]+lzs*xr[60])));
    msp_[69] = std::max(0.9, std::min(1.0, 1.116 + 0.166 * lzs));
    msp_[70] = std::max(1.477 + 0.296 * lzs, std::min(1.6, -0.308 - 1.046 * lzs));
    msp_[70] = std::max(0.8, std::min(0.8 - 2.0 * lzs, msp_[70]));
    msp_[71] = xr[61]+lzs*(xr[62]+lzs*xr[63]);
    msp_[72] = std::max(0.065, 0.0843 - lzs * (0.0475 + 0.0352 * lzs));
    msp_[73] = 0.0736 + lzs * (0.0749 + 0.04426 * lzs);
    if (z < 0.004) msp_[73] = std::min(0.055, msp_[73]);
    msp_[74] = std::max(0.091, std::min(0.121, 0.136 + 0.0352 * lzs));
    msp_[75] = (msp_[64] * std::pow(msp_[70], msp_[66])) /
               (msp_[65] + std::pow(msp_[70], msp_[67]));
    if (msp_[69] > msp_[70]) {
        msp_[69] = msp_[70];
        msp_[74] = msp_[75];
    }

    // Rbeta: msp(77..82) -> msp_[76..81]
    msp_[76] = xr[64]+lzs*(xr[65]+lzs*(xr[66]+lzs*xr[67]));
    msp_[77] = xr[68]+lzs*(xr[69]+lzs*(xr[70]+lzs*xr[71]));
    msp_[78] = xr[72]+lzs*(xr[73]+lzs*(xr[74]+lzs*xr[75]));
    msp_[79] = xr[76]+lzs*(xr[77]+lzs*(xr[78]+lzs*xr[79]));
    msp_[80] = xr[80]+lzs*(xr[81]+lzs*lzs*xr[82]);
    if (z > 0.01) msp_[80] = std::max(msp_[80], 0.95);
    msp_[81] = std::max(1.4, std::min(1.6, 1.6 + lzs * (0.764 + 0.3322 * lzs)));

    // Rgamma: msp(83..89) -> msp_[82..88]
    msp_[82] = std::max(xr[83]+lzs*(xr[84]+lzs*(xr[85]+lzs*xr[86])),
                        xr[95]+lzs*(xr[96]+lzs*xr[97]));
    msp_[83] = std::min(0.0, xr[87]+lzs*(xr[88]+lzs*(xr[89]+lzs*xr[90])));
    msp_[83] = std::max(msp_[83], xr[98]+lzs*(xr[99]+lzs*xr[100]));
    msp_[84] = xr[91]+lzs*(xr[92]+lzs*(xr[93]+lzs*xr[94]));
    msp_[84] = std::max(0.0, std::min(msp_[84], 7.454 + 9.046 * lzs));
    msp_[85] = std::min(xr[101]+lzs*xr[102], std::max(2.0, -13.3 - 18.6 * lzs));
    msp_[86] = std::min(1.5, std::max(0.4, 2.493 + 1.1475 * lzs));
    msp_[87] = std::max(1.0, std::min(1.27, 0.8109 - 0.6282 * lzs));
    msp_[87] = std::max(msp_[87], 0.6355 - 0.4192 * lzs);
    msp_[88] = std::max(5.855420e-02, -0.2711 - lzs * (0.5756 + 0.0838 * lzs));

    // Rhook: msp(90..96) -> msp_[89..95]
    msp_[89] = xr[103]+lzs*(xr[104]+lzs*(xr[105]+lzs*xr[106]));
    msp_[90] = xr[107]+lzs*(xr[108]+lzs*(xr[109]+lzs*xr[110]));
    msp_[91] = xr[111]+lzs*(xr[112]+lzs*(xr[113]+lzs*xr[114]));
    msp_[92] = xr[115]+lzs*(xr[116]+lzs*(xr[117]+lzs*xr[118]));
    msp_[93] = std::min(1.25, std::max(1.1, 1.9848 + lzs * (1.1386 + 0.3564 * lzs)));
    msp_[94] = 0.063 + lzs * (0.0481 + 0.00984 * lzs);
    msp_[95] = std::min(1.3, std::max(0.45, 1.2 + 2.45 * lzs));

    // Lneta: msp(97) -> msp_[96]
    if (z > 0.0009) msp_[96] = 10.0; else msp_[96] = 20.0;

    // Lbgb: gbp(1..8) -> gbp_[0..7]
    gbp_[0] = xg[0]+lzs*(xg[1]+lzs*(xg[2]+lzs*xg[3]));
    gbp_[1] = xg[4]+lzs*(xg[5]+lzs*(xg[6]+lzs*xg[7]));
    gbp_[2] = xg[8]+lzs*(xg[9]+lzs*(xg[10]+lzs*xg[11]));
    gbp_[3] = xg[12]+lzs*(xg[13]+lzs*(xg[14]+lzs*xg[15]));
    gbp_[4] = xg[16]+lzs*(xg[17]+lzs*xg[18]);
    gbp_[5] = xg[19]+lzs*(xg[20]+lzs*xg[21]);
    gbp_[2] = std::pow(gbp_[2], gbp_[5]); // gbp(3) = gbp(3)**gbp(6)
    gbp_[6] = xg[22];
    gbp_[7] = xg[23];

    // Lbagb: gbp(9..16) -> gbp_[8..15]
    gbp_[8]  = xg[24] + lzs*(xg[25] + lzs*xg[26]);
    gbp_[9]  = xg[27] + lzs*(xg[28] + lzs*xg[29]);
    gbp_[10] = 15.0;
    gbp_[11] = xg[30]+lzs*(xg[31]+lzs*(xg[32]+lzs*xg[33]));
    gbp_[12] = xg[34]+lzs*(xg[35]+lzs*(xg[36]+lzs*xg[37]));
    gbp_[13] = xg[38]+lzs*(xg[39]+lzs*(xg[40]+lzs*xg[41]));
    gbp_[14] = xg[42]+lzs*xg[43];
    gbp_[11] = std::pow(gbp_[11], gbp_[14]); // gbp(12) = gbp(12)**gbp(15)
    gbp_[13] = std::pow(gbp_[13], gbp_[14]); // gbp(14) = gbp(14)**gbp(15)
    gbp_[15] = 1.0; // will be reset later

    // Rgb: gbp(17..23) -> gbp_[16..22]
    gbp_[16] = -4.6739 - 0.9394 * lz;
    gbp_[16] = std::pow(10.0, gbp_[16]);
    gbp_[16] = std::max(gbp_[16], -0.04167 + 55.67 * z);
    gbp_[16] = std::min(gbp_[16], 0.4771 - 9329.21 * std::pow(z, 2.94));
    gbp_[17] = std::min(0.54, 0.397 + lzs * (0.28826 + 0.5293 * lzs));
    gbp_[18] = std::max(-0.1451, -2.2794 - lz * (1.5175 + 0.254 * lz));
    gbp_[18] = std::pow(10.0, gbp_[18]);
    if (z > 0.004) {
        gbp_[18] = std::max(gbp_[18], 0.7307 + 14265.1 * std::pow(z, 3.395));
    }
    gbp_[19] = xg[44]+lzs*(xg[45]+lzs*(xg[46]+lzs*(xg[47]+lzs*(xg[48]+lzs*xg[49]))));
    gbp_[20] = xg[50]+lzs*(xg[51]+lzs*(xg[52]+lzs*(xg[53]+lzs*xg[54])));
    gbp_[21] = xg[55]+lzs*(xg[56]+lzs*(xg[57]+lzs*(xg[58]+lzs*(xg[59]+lzs*xg[60]))));
    gbp_[22] = xg[61]+lzs*(xg[62]+lzs*(xg[63]+lzs*(xg[64]+lzs*xg[65])));

    // Ragb: gbp(24..32) -> gbp_[23..31]
    gbp_[23] = std::min(0.99164 - 743.123 * std::pow(z, 2.83),
                        1.0422 + lzs * (0.13156 + 0.045 * lzs));
    gbp_[24] = xg[66]+lzs*(xg[67]+lzs*(xg[68]+lzs*(xg[69]+lzs*(xg[70]+lzs*xg[71]))));
    gbp_[25] = xg[72]+lzs*(xg[73]+lzs*(xg[74]+lzs*(xg[75]+lzs*xg[76])));
    gbp_[26] = xg[77]+lzs*(xg[78]+lzs*(xg[79]+lzs*(xg[80]+lzs*(xg[81]+lzs*xg[82]))));
    gbp_[27] = xg[83]+lzs*(xg[84]+lzs*(xg[85]+lzs*(xg[86]+lzs*xg[87])));
    gbp_[28] = xg[88]+lzs*(xg[89]+lzs*(xg[90]+lzs*(xg[91]+lzs*(xg[92]+lzs*xg[93]))));
    gbp_[29] = xg[94]+lzs*(xg[95]+lzs*(xg[96]+lzs*(xg[97]+lzs*(xg[98]+lzs*xg[99]))));
    double m1 = zpars_[1] - 0.2;
    gbp_[30] = gbp_[28] + gbp_[29] * m1;
    gbp_[31] = std::min(gbp_[24] / std::pow(zpars_[1], gbp_[25]),
                        gbp_[26] / std::pow(zpars_[1], gbp_[27]));

    // Mchei: gbp(33..34) -> gbp_[32..33]
    gbp_[32] = std::pow(xg[100], 4);
    gbp_[33] = xg[101] * 4.0;

    // Mcagb: gbp(35..37) -> gbp_[34..36]
    gbp_[34] = xg[102]+lzs*(xg[103]+lzs*(xg[104]+lzs*xg[105]));
    gbp_[35] = xg[106]+lzs*(xg[107]+lzs*(xg[108]+lzs*xg[109]));
    gbp_[36] = xg[110]+lzs*xg[111];
    gbp_[34] = std::pow(gbp_[34], 4);
    gbp_[35] = gbp_[35] * 4.0;
    gbp_[36] = std::pow(gbp_[36], 4);

    // Lhei: gbp(38..44) -> gbp_[37..43]
    gbp_[37] = xh[0]+lzs*xh[1];
    gbp_[38] = xh[2]+lzs*xh[3];
    gbp_[39] = xh[4];
    gbp_[40] = -1.0; // reset later
    gbp_[41] = xh[5]+lzs*(xh[6]+lzs*xh[7]);
    gbp_[42] = xh[8]+lzs*(xh[9]+lzs*xh[10]);
    gbp_[43] = xh[11]+lzs*(xh[12]+lzs*xh[13]);
    gbp_[41] = gbp_[41] * gbp_[41]; // gbp(42) = gbp(42)**2
    gbp_[43] = gbp_[43] * gbp_[43]; // gbp(44) = gbp(44)**2

    // Lhe: gbp(45..48) -> gbp_[44..47]
    gbp_[44] = xh[14]+lzs*(xh[15]+lzs*xh[16]);
    if (lzs > -1.0) {
        gbp_[45] = 1.0 - xh[18] * std::pow(lzs + 1.0, xh[17]);
    } else {
        gbp_[45] = 1.0;
    }
    gbp_[46] = xh[19]+lzs*(xh[20]+lzs*xh[21]);
    gbp_[47] = xh[22]+lzs*(xh[23]+lzs*xh[24]);
    gbp_[44] = std::pow(gbp_[44], gbp_[47]);
    gbp_[46] = std::pow(gbp_[46], gbp_[47]);
    gbp_[45] = gbp_[45] / std::pow(zpars_[2], 0.1) +
               (gbp_[45] * gbp_[46] - gbp_[44]) /
               std::pow(zpars_[2], gbp_[47] + 0.1);

    // Rmin: gbp(49..53) -> gbp_[48..52]
    gbp_[48] = xh[25]+lzs*(xh[26]+lzs*(xh[27]+lzs*xh[28]));
    gbp_[49] = xh[29]+lzs*(xh[30]+lzs*(xh[31]+lzs*xh[32]));
    gbp_[50] = xh[33]+lzs*(xh[34]+lzs*(xh[35]+lzs*xh[36]));
    gbp_[51] = 5.0 + xh[37] * std::pow(z, xh[38]);
    gbp_[52] = xh[39]+lzs*(xh[40]+lzs*(xh[41]+lzs*xh[42]));
    gbp_[48] = std::pow(gbp_[48], gbp_[52]);
    gbp_[50] = std::pow(gbp_[50], 2.0 * gbp_[52]);

    // The: gbp(54..61) -> gbp_[53..60]
    gbp_[53] = xh[43]+lzs*(xh[44]+lzs*(xh[45]+lzs*xh[46]));
    gbp_[54] = xh[47]+lzs*(xh[48]+lzs*xh[49]);
    gbp_[54] = std::max(gbp_[54], 1.0);
    gbp_[55] = xh[50];
    gbp_[56] = -1.0; // reset later
    gbp_[57] = xh[51]+lzs*(xh[52]+lzs*(xh[53]+lzs*xh[54]));
    gbp_[58] = xh[55]+lzs*(xh[56]+lzs*(xh[57]+lzs*xh[58]));
    gbp_[59] = xh[59]+lzs*(xh[60]+lzs*(xh[61]+lzs*xh[62]));
    gbp_[60] = xh[63]+lzs*xh[64];
    gbp_[57] = std::pow(gbp_[57], gbp_[60]);
    gbp_[59] = std::pow(gbp_[59], 5);

    // Tbl: gbp(62..68) -> gbp_[61..67]
    double dum1 = zpars_[1] / zpars_[2];
    gbp_[61] = xh[65]+lzs*xh[66];
    gbp_[61] = -gbp_[61] * std::log10(dum1);
    gbp_[62] = xh[67];
    if (lzd > 0.0) {
        gbp_[63] = 1.0 - lzd * (xh[68] + lzd * (xh[69] + lzd * xh[70]));
    } else {
        gbp_[63] = 1.0;
    }
    gbp_[64] = 1.0 - gbp_[63] * std::pow(dum1, gbp_[62]);
    gbp_[65] = 1.0 - lzd * (xh[76] + lzd * (xh[77] + lzd * xh[78]));
    gbp_[66] = xh[71] + lzs * (xh[72] + lzs * (xh[73] + lzs * xh[74]));
    gbp_[67] = xh[75];

    // Lzahb: gbp(69..74) -> gbp_[68..73]
    gbp_[68] = xh[79] + lzs * (xh[80] + lzs * xh[81]);
    gbp_[69] = xh[82] + lzs * (xh[83] + lzs * xh[84]);
    gbp_[70] = 15.0;
    gbp_[71] = xh[85];
    gbp_[72] = xh[86];

    // Rzahb: gbp(75..77) -> gbp_[74..76]
    gbp_[74] = xh[87] + lzs * (xh[88] + lzs * (xh[89] + lzs * xh[90]));
    gbp_[75] = xh[91] + lzs * (xh[92] + lzs * (xh[93] + lzs * xh[94]));
    gbp_[76] = xh[95] + lzs * (xh[96] + lzs * (xh[97] + lzs * xh[98]));

    // Finish Lbagb: gbp(16) = lbagbf(zpars(2), 0.0)
    double mhefl = 0.0;
    double lx = lbagbf(zpars_[1], mhefl);
    gbp_[15] = lx;

    // Finish LHeI
    double lhefl = lheif(zpars_[1], mhefl);
    gbp_[40] = (gbp_[37] * std::pow(zpars_[1], gbp_[38]) - lhefl) /
               (std::exp(zpars_[1] * gbp_[39]) * lhefl);

    // Finish THe
    double dum0 = 0.0;
    double thefl = thef(zpars_[1], dum0, mhefl) * tbgbf(zpars_[1]);
    gbp_[56] = (thefl - gbp_[53]) / (gbp_[53] * std::exp(gbp_[55] * zpars_[1]));

    // Finish Tblf
    double rb = ragbf(zpars_[2], lheif(zpars_[2], zpars_[1]), mhefl);
    double rr = 1.0 - rminf(zpars_[2]) / rb;
    rr = std::max(rr, 1.0e-12);
    gbp_[65] = gbp_[65] / (std::pow(zpars_[2], gbp_[66]) * std::pow(rr, gbp_[67]));

    // Finish Lzahb: gbp(74) -> gbp_[73]
    gbp_[73] = lhefl * lhef(zpars_[1]);

    // Call star to get GB params for zpars(2)
    int kw = 0;
    double tm = 0.0, tn = 0.0;
    double tscls[20], lums[10], GB[10];
    std::fill(tscls, tscls + 20, 0.0);
    std::fill(lums, lums + 10, 0.0);
    std::fill(GB, GB + 10, 0.0);
    star(kw, zpars_[1], zpars_[1], tm, tn, tscls, lums, GB);
    zpars_[8] = mcgbf(lums[2], GB, lums[5]);
    zpars_[9] = mcgbf(lums[3], GB, lums[5]);

    // hydrogen and helium abundances
    zpars_[10] = 0.76 - 3.0 * z;
    zpars_[11] = 0.24 + 2.0 * z;

    // constant for low-mass CHeB stars
    zpars_[12] = rminf(zpars_[1]) /
                 rgbf(zpars_[1], lzahbf(zpars_[1], zpars_[8], zpars_[1]));

    zpars_[13] = std::pow(z, 0.4);
}

// =========================================================================
// Fitting formulae - port of zfuncs.f
// All use 0-based msp_[] and gbp_[] arrays.
// Fortran a(N) -> msp_[N-1] or gbp_[N-1] depending on common block.
// =========================================================================

double SSEContext::lzamsf(double m) const {
    double mx = std::sqrt(m);
    return (msp_[0]*std::pow(m,5)*mx + msp_[1]*std::pow(m,11)) /
           (msp_[2] + std::pow(m,3) + msp_[3]*std::pow(m,5) +
            msp_[4]*std::pow(m,7) + msp_[5]*std::pow(m,8) +
            msp_[6]*std::pow(m,9)*mx);
}

double SSEContext::rzamsf(double m) const {
    double mx = std::sqrt(m);
    return ((msp_[7]*m*m + msp_[8]*std::pow(m,6))*mx +
             msp_[9]*std::pow(m,11) +
            (msp_[10] + msp_[11]*mx)*std::pow(m,19)) /
           (msp_[12] + msp_[13]*m*m +
            (msp_[14]*std::pow(m,8) + std::pow(m,18) +
             msp_[15]*std::pow(m,19))*mx);
}

double SSEContext::tbgbf(double m) const {
    return (msp_[16] + msp_[17]*std::pow(m,4) +
            msp_[18]*std::pow(m,5.5) + std::pow(m,7)) /
           (msp_[19]*m*m + msp_[20]*std::pow(m,7));
}

double SSEContext::thookf(double m) const {
    double val = 1.0 - 0.01 * std::max(msp_[21]/std::pow(m,msp_[22]),
                                        msp_[23]+msp_[24]/std::pow(m,msp_[25]));
    return std::max(val, 0.5);
}

double SSEContext::ltmsf(double m) const {
    return (msp_[26]*std::pow(m,3) + msp_[27]*std::pow(m,4) +
            msp_[28]*std::pow(m,msp_[31]+1.8)) /
           (msp_[29] + msp_[30]*std::pow(m,5) + std::pow(m,msp_[31]));
}

double SSEContext::lalphf(double m) const {
    double mcut = 2.0;
    if (m >= mcut) {
        return (msp_[32] + msp_[33]*std::pow(m,msp_[35])) /
               (std::pow(m,0.4) + msp_[34]*std::pow(m,1.9));
    } else {
        if (m <= 0.5)
            return msp_[38];
        else if (m <= 0.7)
            return msp_[38] + ((0.3 - msp_[38])/0.2) * (m - 0.5);
        else if (m <= msp_[36])
            return 0.3 + ((msp_[39]-0.3)/(msp_[36]-0.7)) * (m - 0.7);
        else if (m <= msp_[37])
            return msp_[39] + ((msp_[40]-msp_[39])/(msp_[37]-msp_[36])) * (m - msp_[36]);
        else
            return msp_[40] + ((msp_[41]-msp_[40])/(mcut-msp_[37])) * (m - msp_[37]);
    }
}

double SSEContext::lbetaf(double m) const {
    double val = msp_[42] - msp_[43]*std::pow(m,msp_[44]);
    val = std::max(val, 0.0);
    if (m > msp_[45] && val > 0.0) {
        double a1 = msp_[42] - msp_[43]*std::pow(msp_[45],msp_[44]);
        val = a1 - 10.0*a1*(m - msp_[45]);
        val = std::max(val, 0.0);
    }
    return val;
}

double SSEContext::lnetaf(double m) const {
    double val;
    if (m <= 1.0)
        val = 10.0;
    else if (m >= 1.1)
        val = 20.0;
    else
        val = 10.0 + 100.0*(m - 1.0);
    return std::min(val, msp_[96]);
}

double SSEContext::lhookf(double m, double mhook) const {
    if (m <= mhook)
        return 0.0;
    else if (m >= msp_[50]) // a(51)
        return std::min(msp_[46]/std::pow(m,msp_[47]),
                        msp_[48]/std::pow(m,msp_[49]));
    else {
        double a2 = std::min(msp_[46]/std::pow(msp_[50],msp_[47]),
                             msp_[48]/std::pow(msp_[50],msp_[49]));
        return a2 * std::pow((m-mhook)/(msp_[50]-mhook), 0.4);
    }
}

double SSEContext::rtmsf(double m) const {
    double m2 = msp_[61] + 0.1;
    if (m <= msp_[61]) {
        double rchk = 1.5 * rzamsf(m);
        return std::max(rchk, (msp_[51] + msp_[52]*std::pow(m,msp_[54])) /
                              (msp_[53] + std::pow(m,msp_[55])));
    } else if (m >= m2) {
        return (msp_[56]*std::pow(m,3) + msp_[57]*std::pow(m,msp_[60]) +
                msp_[58]*std::pow(m,msp_[60]+1.5)) /
               (msp_[59] + std::pow(m,5));
    } else {
        return msp_[62] + ((msp_[63] - msp_[62])/0.1) * (m - msp_[61]);
    }
}

double SSEContext::ralphf(double m) const {
    if (m <= 0.5)
        return msp_[72]; // a(73)
    else if (m <= 0.65)
        return msp_[72] + ((msp_[73] - msp_[72])/0.15) * (m - 0.5);
    else if (m <= msp_[69]) // a(70)
        return msp_[73] + ((msp_[74]-msp_[73])/(msp_[69]-0.65)) * (m - 0.65);
    else if (m <= msp_[70]) // a(71)
        return msp_[74] + ((msp_[75] - msp_[74])/(msp_[70] - msp_[69])) * (m - msp_[69]);
    else if (m <= msp_[71]) // a(72)
        return (msp_[64]*std::pow(m,msp_[66])) / (msp_[65] + std::pow(m,msp_[67]));
    else {
        double a5 = (msp_[64]*std::pow(msp_[71],msp_[66])) /
                    (msp_[65] + std::pow(msp_[71],msp_[67]));
        return a5 + msp_[68]*(m - msp_[71]);
    }
}

double SSEContext::rbetaf(double m) const {
    double m2v = 2.0, m3 = 16.0;
    double val;
    if (m <= 1.0)
        val = 1.06;
    else if (m <= msp_[81]) // a(82)
        val = 1.06 + ((msp_[80]-1.06)/(msp_[81]-1.0)) * (m-1.0);
    else if (m <= m2v) {
        double b2 = (msp_[76]*std::pow(m2v,3.5)) / (msp_[77] + std::pow(m2v,msp_[78]));
        val = msp_[80] + ((b2-msp_[80])/(m2v-msp_[81])) * (m-msp_[81]);
    } else if (m <= m3)
        val = (msp_[76]*std::pow(m,3.5)) / (msp_[77] + std::pow(m,msp_[78]));
    else {
        double b3 = (msp_[76]*std::pow(m3,3.5)) / (msp_[77] + std::pow(m3,msp_[78]));
        val = b3 + msp_[79]*(m - m3);
    }
    return val - 1.0;
}

double SSEContext::rgammf(double m) const {
    double m1 = 1.0;
    if (m > (msp_[87]+0.1)) // a(88)+0.1
        return 0.0;
    double b1 = std::max(0.0, msp_[82] + msp_[83]*std::pow(std::abs(m1-msp_[84]),msp_[85]));
    double val;
    if (m <= m1) {
        val = msp_[82] + msp_[83]*std::pow(std::abs(m-msp_[84]),msp_[85]);
    } else if (m <= msp_[87]) { // a(88)
        val = b1 + (msp_[88] - b1)*std::pow((m - m1)/(msp_[87] - m1), msp_[86]);
    } else {
        if (msp_[87] > m1) b1 = msp_[88];
        val = b1 - 10.0*b1*(m - msp_[87]);
    }
    return std::max(val, 0.0);
}

double SSEContext::rhookf(double m, double mhook) const {
    if (m <= mhook)
        return 0.0;
    else if (m <= msp_[93]) { // a(94)
        return msp_[94] * std::sqrt((m-mhook)/(msp_[93]-mhook));
    } else if (m <= 2.0) {
        double m2 = 2.0;
        double b2 = (msp_[89] + msp_[90]*std::pow(m2,3.5)) /
                    (msp_[91]*std::pow(m2,3) + std::pow(m2,msp_[92])) - 1.0;
        return msp_[94] + (b2-msp_[94])*std::pow((m-msp_[93])/(m2-msp_[93]), msp_[95]);
    } else {
        return (msp_[89] + msp_[90]*std::pow(m,3.5)) /
               (msp_[91]*std::pow(m,3) + std::pow(m,msp_[92])) - 1.0;
    }
}

// GB functions using gbp_ array (Fortran a(N) -> gbp_[N-1])

double SSEContext::lbgbf(double m) const {
    return (gbp_[0]*std::pow(m,gbp_[4]) + gbp_[1]*std::pow(m,gbp_[7])) /
           (gbp_[2] + gbp_[3]*std::pow(m,gbp_[6]) + std::pow(m,gbp_[5]));
}

double SSEContext::lbagbf(double m, double mhefl) const {
    double a4 = (gbp_[8]*std::pow(mhefl,gbp_[9]) - gbp_[15]) /
                (std::exp(mhefl*gbp_[10]) * gbp_[15]);
    if (m < mhefl) {
        return gbp_[8]*std::pow(m,gbp_[9]) / (1.0 + a4*std::exp(m*gbp_[10]));
    } else {
        return (gbp_[11] + gbp_[12]*std::pow(m,gbp_[14]+1.8)) /
               (gbp_[13] + std::pow(m,gbp_[14]));
    }
}

double SSEContext::rgbf(double m, double lum) const {
    double a1 = std::min(gbp_[19]/std::pow(m,gbp_[20]),
                         gbp_[21]/std::pow(m,gbp_[22]));
    return a1*(std::pow(lum,gbp_[17]) + gbp_[16]*std::pow(lum,gbp_[18]));
}

double SSEContext::ragbf(double m, double lum, double mhelf) const {
    double m1 = mhelf - 0.2;
    double xx;
    if (m >= mhelf)
        xx = gbp_[23];
    else if (m >= m1)
        xx = 1.0 + 5.0*(gbp_[23]-1.0)*(m-m1);
    else
        xx = 1.0;
    double a4 = xx * gbp_[18];
    double a1;
    if (m <= m1)
        a1 = gbp_[28] + gbp_[29]*m;
    else if (m >= mhelf)
        a1 = std::min(gbp_[24]/std::pow(m,gbp_[25]),
                      gbp_[26]/std::pow(m,gbp_[27]));
    else
        a1 = gbp_[30] + 5.0*(gbp_[31]-gbp_[30])*(m-m1);
    return a1*(std::pow(lum,gbp_[17]) + gbp_[16]*std::pow(lum,a4));
}

double SSEContext::mctmsf(double m) const {
    double m525 = std::pow(m, 5.25);
    return (1.586 + m525) / (2.434 + 1.02*m525);
}

double SSEContext::mcheif(double m, double mhefl, double mchefl) const {
    double mcbagb = mcagbf(m);
    double a3 = std::pow(mchefl,4) - gbp_[32]*std::pow(mhefl,gbp_[33]);
    return std::min(0.95*mcbagb, std::pow(a3 + gbp_[32]*std::pow(m,gbp_[33]), 0.25));
}

double SSEContext::mcagbf(double m) const {
    return std::pow(gbp_[36] + gbp_[34]*std::pow(m,gbp_[35]), 0.25);
}

double SSEContext::mbagbf(double mc) const {
    double mc4 = std::pow(mc, 4);
    if (mc4 > gbp_[36])
        return std::pow((mc4 - gbp_[36])/gbp_[34], 1.0/gbp_[35]);
    else
        return 0.0;
}

double SSEContext::mcgbtf(double t, double A, const double GB[10],
                          double tinf1, double tinf2, double tx) const {
    if (t <= tx) {
        return std::pow((GB[4]-1.0)*A*GB[3]*(tinf1 - t), 1.0/(1.0-GB[4]));
    } else {
        return std::pow((GB[5]-1.0)*A*GB[2]*(tinf2 - t), 1.0/(1.0-GB[5]));
    }
}

double SSEContext::lgbtf(double t, double A, const double GB[10],
                         double tinf1, double tinf2, double tx) const {
    if (t <= tx) {
        return GB[3] * std::pow((GB[4]-1.0)*A*GB[3]*(tinf1 - t),
                                GB[4]/(1.0-GB[4]));
    } else {
        return GB[2] * std::pow((GB[5]-1.0)*A*GB[2]*(tinf2 - t),
                                GB[5]/(1.0-GB[5]));
    }
}

double SSEContext::mcgbf(double lum, const double GB[10], double lx) const {
    if (lum <= lx) {
        return std::pow(lum/GB[3], 1.0/GB[4]);
    } else {
        return std::pow(lum/GB[2], 1.0/GB[5]);
    }
}

double SSEContext::lmcgbf(double mc, const double GB[10]) const {
    if (mc <= GB[6]) {
        return GB[3] * std::pow(mc, GB[4]);
    } else {
        return GB[2] * std::pow(mc, GB[5]);
    }
}

double SSEContext::lheif(double m, double mhefl) const {
    if (m < mhefl) {
        return gbp_[37]*std::pow(m,gbp_[38]) /
               (1.0 + gbp_[40]*std::exp(m*gbp_[39]));
    } else {
        return (gbp_[41] + gbp_[42]*std::pow(m,3.8)) /
               (gbp_[43] + m*m);
    }
}

double SSEContext::lhef(double m) const {
    return (gbp_[44] + gbp_[45]*std::pow(m,gbp_[47]+0.1)) /
           (gbp_[46] + std::pow(m,gbp_[47]));
}

double SSEContext::rminf(double m) const {
    double mx = std::pow(m, gbp_[52]);
    return (gbp_[48]*m + std::pow(gbp_[49]*m,gbp_[51])*mx) /
           (gbp_[50] + mx);
}

double SSEContext::thef(double m, double mc, double mhefl) const {
    if (m <= mhefl) {
        double mm = std::max((mhefl - m)/(mhefl - mc), 1.0e-12);
        return (gbp_[53] + (themsf(mc) - gbp_[53])*std::pow(mm,gbp_[54])) *
               (1.0 + gbp_[56]*std::exp(m*gbp_[55]));
    } else {
        double mm = std::pow(m, 5);
        return (gbp_[57]*std::pow(m,gbp_[60]) + gbp_[58]*mm) /
               (gbp_[59] + mm);
    }
}

double SSEContext::tblf(double m, double mhefl, double mfgb) const {
    double mr = mhefl / mfgb;
    double val;
    if (m <= mfgb) {
        double m1 = m / mfgb;
        double m2v = std::log10(m1) / std::log10(mr);
        m2v = std::max(m2v, 1.0e-12);
        val = gbp_[63]*std::pow(m1,gbp_[62]) + gbp_[64]*std::pow(m2v,gbp_[61]);
    } else {
        double r1 = 1.0 - rminf(m) / ragbf(m, lheif(m, mhefl), mhefl);
        r1 = std::max(r1, 1.0e-12);
        val = gbp_[65]*std::pow(m,gbp_[66])*std::pow(r1,gbp_[67]);
    }
    val = std::min(1.0, std::max(0.0, val));
    if (val < 1.0e-10) val = 0.0;
    return val;
}

double SSEContext::lzahbf(double m, double mc, double mhefl) const {
    double a5 = lzhef(mc);
    double a4 = (gbp_[68] + a5 - gbp_[73]) /
                ((gbp_[73] - a5) * std::exp(gbp_[70]*mhefl));
    double mm = std::max((m-mc)/(mhefl - mc), 1.0e-12);
    return a5 + (1.0 + gbp_[71])*gbp_[68]*std::pow(mm,gbp_[69]) /
           ((1.0 + gbp_[71]*std::pow(mm,gbp_[72])) *
            (1.0 + a4*std::exp(m*gbp_[70])));
}

double SSEContext::rzahbf(double m, double mc, double mhefl) const {
    double rx = rzhef(mc);
    double ry = rgbf(m, lzahbf(m, mc, mhefl));
    double mm = std::max((m-mc)/(mhefl - mc), 1.0e-12);
    double f = (1.0 + gbp_[75])*std::pow(mm,gbp_[74]) /
               (1.0 + gbp_[75]*std::pow(mm,gbp_[76]));
    return (1.0 - f)*rx + f*ry;
}

double SSEContext::lzhef(double m) const {
    double m15 = m * std::sqrt(m);
    return 1.5262e+04 * std::pow(m, 10.25) /
           (0.0469 + std::pow(m,6)*(31.18 + m15*(29.54 + m15)));
}

double SSEContext::rzhef(double m) const {
    return 0.2391*std::pow(m,4.6) / (0.0065 + (0.162+m)*std::pow(m,3));
}

double SSEContext::themsf(double m) const {
    return (0.4129 + 18.81*std::pow(m,4) + 1.853*std::pow(m,6)) /
           std::pow(m, 6.5);
}

double SSEContext::rhehgf(double m, double lum, double rx, double lx) const {
    double cm = 2.0e-03*std::pow(m,2.5) / (2.0 + std::pow(m,5));
    return rx*std::pow(lum/lx, 0.2) + 0.02*(std::exp(cm*lum) - std::exp(cm*lx));
}

double SSEContext::rhegbf(double lum) const {
    return 0.08 * std::pow(lum, 0.75);
}

double SSEContext::lpertf(double m, double mew) const {
    double b = 0.002 * std::max(1.0, 2.5/m);
    double c = 3.0;
    return ((1.0 + std::pow(b,c)) * std::pow(mew/b,c)) / (1.0 + std::pow(mew/b,c));
}

double SSEContext::rpertf(double m, double mew, double r, double rc) const {
    if (mew <= 0.0) return 0.0;
    double a = 0.1;
    double b = 0.006 * std::max(1.0, 2.5/m);
    double c = 3.0;
    double q = std::log(r/rc);
    double fac = a/q;
    double facmax = -14.0/std::log10(mew);
    fac = std::min(fac, facmax);
    return ((1.0 + std::pow(b,c)) * std::pow(mew/b,c) * std::pow(mew,fac)) /
           (1.0 + std::pow(mew/b,c));
}

// =========================================================================
// mlwind - port of mlwind.f with Vink 2001 extensions (Kamlah/DRAGON-II)
// =========================================================================
double SSEContext::mlwind(int kw, double lum, double r, double mt,
                          double mc, double rl, double z) const {
    const double lum0 = 7.0e+04, kap = -0.5;
    const int mdflag = static_cast<int>(cfg_.wind);
    double dms = 0.0, dml = 0.0;

    if (mdflag <= 2) {
        // ---- HURLEY / HURLEY_LBV (mdflag 1-2) ----
        // Nieuwenhuijzen & de Jager for massive stars
        if (lum > 4000.0) {
            double x = std::min(1.0, (lum-4000.0)/500.0);
            dms = 9.6e-15 * x * std::pow(r,0.81) * std::pow(lum,1.24) * std::pow(mt,0.16);
            dms = dms * std::sqrt(z/0.02);
        }
        if (kw >= 2 && kw <= 9) {
            // Reimers mass loss
            dml = cfg_.neta * 4.0e-13 * r * lum / mt;
            if (rl > 0.0) dml = dml * (1.0 + cfg_.bwind * std::pow(std::min(0.5, r/rl), 6));
            // Vassiliadis & Wood for AGB
            if (kw == 5 || kw == 6) {
                double p0 = -2.07 - 0.9*std::log10(mt) + 1.94*std::log10(r);
                p0 = std::pow(10.0, p0);
                p0 = std::min(p0, 2000.0);
                double dmt = -11.4 + 0.0125*(p0 - 100.0*std::max(mt-2.5, 0.0));
                dmt = std::pow(10.0, dmt);
                dmt = 1.0 * std::min(dmt, 1.36e-09*lum);
                dml = std::max(dml, dmt);
            }
            if (kw > 6) {
                dms = std::max(dml, 1.0e-13 * cfg_.hewind * std::pow(lum, 1.5));
            } else {
                dms = std::max(dml, dms);
                double mew = ((mt-mc)/mt) * std::min(5.0, std::max(1.2, std::pow(lum/lum0, kap)));
                // reduced WR-like mass loss
                if (mew < 1.0) {
                    dml = 1.0e-13 * std::pow(lum, 1.5) * (1.0 - mew);
                    dms = std::max(dml, dms);
                }
                // LBV-like mass loss
                double x = 1.0e-5 * r * std::sqrt(lum);
                if (lum > 6.0e+05 && x > 1.0) {
                    if (mdflag == 1) {
                        // Original Hurley: cubic ramp
                        dml = 0.1 * std::pow(x-1.0, 3) * (lum/6.0e+05 - 1.0);
                    } else {
                        // HURLEY_LBV: flat rate
                        dml = 1.0e-4 * cfg_.flbv;
                    }
                    dms = dms + dml;
                }
            }
        }
    } else {
        // ---- VINK / VINK_NO_BISTAB (mdflag 3-4) ----
        if (kw >= 2 && kw <= 9) {
            // Reimers mass loss (always present as floor)
            dml = cfg_.neta * 4.0e-13 * r * lum / mt;
            if (rl > 0.0) dml = dml * (1.0 + cfg_.bwind * std::pow(std::min(0.5, r/rl), 6));
            // Vassiliadis & Wood for AGB
            if (kw == 5 || kw == 6) {
                double p0 = -2.07 - 0.9*std::log10(mt) + 1.94*std::log10(r);
                p0 = std::pow(10.0, p0);
                p0 = std::min(p0, 2000.0);
                double dmt = -11.4 + 0.0125*(p0 - 100.0*std::max(mt-2.5, 0.0));
                dmt = std::pow(10.0, dmt);
                dmt = 1.0 * std::min(dmt, 1.36e-09*lum);
                dml = std::max(dml, dmt);
            }
            if (kw > 6) {
                // Naked helium star WR: Z-dependent (Vink prescription)
                double dms_wr = 1.0e-13 * std::pow(lum, 1.5) * std::pow(z/0.02, 0.86);
                dms = std::max(dml, dms_wr);
            } else {
                // Effective temperature for Vink regime boundaries
                double teff = 5778.0 * std::sqrt(std::sqrt(lum)) / std::sqrt(r);
                double logl5 = std::log10(lum / 1.0e5);
                double logm30 = std::log10(mt / 30.0);
                double logz02 = std::log10(z / 0.02);

                // LBV first (overrides everything if triggered)
                double xlbv = 1.0e-5 * r * std::sqrt(lum);
                if (lum > 6.0e+05 && xlbv > 1.0) {
                    dms = 1.0e-4 * cfg_.flbv;
                } else if (mdflag == 3 && teff >= 12500.0 && teff <= 25000.0) {
                    // Vink cool-side (bi-stability jump)
                    double logt2 = std::log10(teff / 20000.0);
                    double logdm = -6.688 + 2.210*logl5 - 1.339*logm30
                                   - 1.601*std::log10(0.65) + 0.85*logz02
                                   + 1.07*logt2;
                    dms = std::pow(10.0, logdm);
                } else if (teff > 12500.0) {
                    // Vink hot-side
                    double t40 = std::log10(teff / 40000.0);
                    double logdm = -6.697 + 2.194*logl5 - 1.313*logm30
                                   - 1.226*std::log10(1.3) + 0.85*logz02
                                   + 0.933*t40*(1.0 - 11.704*t40);
                    dms = std::pow(10.0, logdm);
                } else {
                    // Cool stars (Teff < 12500K): NdJ fallback
                    if (lum > 4000.0) {
                        double x = std::min(1.0, (lum-4000.0)/500.0);
                        dms = 9.6e-15 * x * std::pow(r,0.81) * std::pow(lum,1.24) * std::pow(mt,0.16);
                        dms = dms * std::sqrt(z/0.02);
                    }
                }
                dms = std::max(dml, dms);

                // Reduced WR-like mass loss for thin envelopes
                double mew = ((mt-mc)/mt) * std::min(5.0, std::max(1.2, std::pow(lum/lum0, kap)));
                if (mew < 1.0) {
                    double dms_wr = 1.0e-13 * std::pow(lum, 1.5) * std::pow(z/0.02, 0.86) * (1.0 - mew);
                    dms = std::max(dms_wr, dms);
                }
            }
        } else {
            // Non-evolved stars (kw 0-1): NdJ only
            if (lum > 4000.0) {
                double x = std::min(1.0, (lum-4000.0)/500.0);
                dms = 9.6e-15 * x * std::pow(r,0.81) * std::pow(lum,1.24) * std::pow(mt,0.16);
                dms = dms * std::sqrt(z/0.02);
            }
        }
    }

    return dms;
}

// =========================================================================
// deltat - port of deltat.f
// =========================================================================
void SSEContext::deltat(int kw, double age, double tm, double tn,
                        const double tscls[20], double& dt, double& dtr) const {
    if (kw <= 1) {
        dt = cfg_.pts1 * tm;
        dtr = tm - age;
    } else if (kw == 2) {
        dt = cfg_.pts1 * (tscls[0] - tm);
        dtr = tscls[0] - age;
    } else if (kw == 3) {
        if (age < tscls[5])
            dt = cfg_.pts2 * (tscls[3] - age);
        else
            dt = cfg_.pts2 * (tscls[4] - age);
        dtr = std::min(tscls[1], tn) - age;
    } else if (kw == 4) {
        dt = cfg_.pts2 * tscls[2];
        dtr = std::min(tn, tscls[1] + tscls[2]) - age;
    } else if (kw == 5) {
        if (age < tscls[8])
            dt = cfg_.pts3 * (tscls[6] - age);
        else
            dt = cfg_.pts3 * (tscls[7] - age);
        dtr = std::min(tn, tscls[12]) - age;
    } else if (kw == 6) {
        if (age < tscls[11])
            dt = cfg_.pts3 * (tscls[9] - age);
        else
            dt = cfg_.pts3 * (tscls[10] - age);
        dt = std::min(dt, 0.005);
        dtr = tn - age;
    } else if (kw == 7) {
        dt = cfg_.pts1 * tm;
        dtr = tm - age;
    } else if (kw == 8 || kw == 9) {
        if (age < tscls[5])
            dt = cfg_.pts2 * (tscls[3] - age);
        else
            dt = cfg_.pts2 * (tscls[4] - age);
        dtr = tn - age;
    } else {
        dt = std::max(0.1, dt*10.0);
        dt = std::min(dt, 5.0e+02);
        dtr = dt;
    }
}

// =========================================================================
// star - port of star.f
// Computes characteristic luminosities, timescales, and GB parameters.
// =========================================================================
void SSEContext::star(int& kw, double mass, double mt, double& tm, double& tn,
                     double tscls[20], double lums[10], double GB[10]) const {
    const double mch = 1.44;
    double mass0 = mass;
    if (mass0 > 100.0) mass = 100.0;

    double tgb = 0.0, tbagb, mcmax, mc1, mc2, mcbagb, dx, lambda;

    if (kw >= 7 && kw <= 9) goto label_90;
    if (kw >= 10) goto label_95;

    // MS and BGB times
    tscls[0] = tbgbf(mass);
    tm = std::max(zpars_[7], thookf(mass)) * tscls[0];

    // ZAMS and TMS luminosity
    lums[0] = lzamsf(mass);
    lums[1] = ltmsf(mass);

    // GB parameters
    GB[0] = std::max(-4.8, std::min(-5.7+0.8*mass, -4.1+0.14*mass));
    GB[0] = std::pow(10.0, GB[0]);
    GB[1] = 1.27e-05;
    GB[7] = 8.0e-05;
    GB[2] = std::max(3.0e+04, 500.0 + 1.75e+04*std::pow(mass,0.6));
    if (mass <= 2.0) {
        GB[3] = zpars_[5];
        GB[4] = 6.0;
        GB[5] = 3.0;
    } else if (mass < 2.5) {
        dx = zpars_[5] - (0.975*zpars_[5] - 0.18*2.5);
        GB[3] = zpars_[5] - dx*(mass - 2.0)/0.5;
        GB[4] = 6.0 - (mass - 2.0)/0.5;
        GB[5] = 3.0 - (mass - 2.0)/0.5;
    } else {
        GB[3] = std::max(-1.0, 0.5*zpars_[5] - 0.06*mass);
        GB[3] = std::max(GB[3], 0.975*zpars_[5] - 0.18*mass);
        GB[4] = 5.0;
        GB[5] = 2.0;
    }
    GB[3] = std::pow(10.0, GB[3]);
    GB[6] = std::pow(GB[2]/GB[3], 1.0/(GB[4]-GB[5]));
    lums[5] = GB[3]*std::pow(GB[6],GB[4]);

    // HeI ignition luminosity
    lums[3] = lheif(mass, zpars_[1]);
    lums[6] = lbagbf(mass, zpars_[1]);

    if (mass < 0.1 && kw <= 1) {
        tscls[1] = 1.1*tscls[0];
        tscls[2] = 0.1*tscls[0];
        lums[2] = lbgbf(mass);
        goto label_96;
    }

    if (mass <= zpars_[2]) {
        lums[2] = lbgbf(mass);
        // GB timescales
        tscls[3] = tscls[0] + (1.0/((GB[4]-1.0)*GB[0]*GB[3])) *
                   std::pow(GB[3]/lums[2], (GB[4]-1.0)/GB[4]);
        tscls[5] = tscls[3] - (tscls[3] - tscls[0]) *
                   std::pow(lums[2]/lums[5], (GB[4]-1.0)/GB[4]);
        tscls[4] = tscls[5] + (1.0/((GB[5]-1.0)*GB[0]*GB[2])) *
                   std::pow(GB[2]/lums[5], (GB[5]-1.0)/GB[5]);
        // He ignition time
        if (lums[3] <= lums[5]) {
            tscls[1] = tscls[3] - (1.0/((GB[4]-1.0)*GB[0]*GB[3])) *
                       std::pow(GB[3]/lums[3], (GB[4]-1.0)/GB[4]);
        } else {
            tscls[1] = tscls[4] - (1.0/((GB[5]-1.0)*GB[0]*GB[2])) *
                       std::pow(GB[2]/lums[3], (GB[5]-1.0)/GB[5]);
        }
        tgb = tscls[1] - tscls[0];
        if (mass <= zpars_[1]) {
            mc1 = mcgbf(lums[3], GB, lums[5]);
            mc2 = mcagbf(mass);
            lums[4] = lzahbf(mass, mc1, zpars_[1]);
            tscls[2] = thef(mass, mc1, zpars_[1]);
        } else {
            lums[4] = lhef(mass) * lums[3];
            tscls[2] = thef(mass, 1.0, zpars_[1]) * tscls[0];
        }
    } else {
        tscls[1] = tscls[0];
        tscls[2] = thef(mass, 1.0, zpars_[1]) * tscls[0];
        lums[4] = lums[6];
        lums[2] = lums[3];
    }

    // Core mass at BGB
    if (mass <= zpars_[1]) {
        GB[8] = mcgbf(lums[2], GB, lums[5]);
    } else if (mass <= zpars_[2]) {
        GB[8] = mcheif(mass, zpars_[1], zpars_[8]);
    } else {
        GB[8] = mcheif(mass, zpars_[1], zpars_[9]);
    }

    // FAGB time parameters
    tbagb = tscls[1] + tscls[2];
    tscls[6] = tbagb + (1.0/((GB[4]-1.0)*GB[7]*GB[3])) *
               std::pow(GB[3]/lums[6], (GB[4]-1.0)/GB[4]);
    tscls[8] = tscls[6] - (tscls[6] - tbagb) *
               std::pow(lums[6]/lums[5], (GB[4]-1.0)/GB[4]);
    tscls[7] = tscls[8] + (1.0/((GB[5]-1.0)*GB[7]*GB[2])) *
               std::pow(GB[2]/lums[5], (GB[5]-1.0)/GB[5]);

    // Find Ltp and ttp
    mcbagb = mcagbf(mass);
    mc1 = mcbagb;
    if (mc1 >= 0.8 && mc1 < 2.25)
        mc1 = 0.44*mc1 + 0.448;
    lums[7] = lmcgbf(mc1, GB);
    if (mc1 <= GB[6]) {
        tscls[12] = tscls[6] - (1.0/((GB[4]-1.0)*GB[7]*GB[3])) *
                    std::pow(mc1, 1.0-GB[4]);
    } else {
        tscls[12] = tscls[7] - (1.0/((GB[5]-1.0)*GB[7]*GB[2])) *
                    std::pow(mc1, 1.0-GB[5]);
    }

    // SAGB time parameters
    if (mc1 <= GB[6]) {
        tscls[9] = tscls[12] + (1.0/((GB[4]-1.0)*GB[1]*GB[3])) *
                   std::pow(GB[3]/lums[7], (GB[4]-1.0)/GB[4]);
        tscls[11] = tscls[9] - (tscls[9] - tscls[12]) *
                    std::pow(lums[7]/lums[5], (GB[4]-1.0)/GB[4]);
        tscls[10] = tscls[11] + (1.0/((GB[5]-1.0)*GB[1]*GB[2])) *
                    std::pow(GB[2]/lums[5], (GB[5]-1.0)/GB[5]);
    } else {
        tscls[9] = tscls[6];
        tscls[11] = tscls[8];
        tscls[10] = tscls[12] + (1.0/((GB[5]-1.0)*GB[1]*GB[2])) *
                    std::pow(GB[2]/lums[7], (GB[5]-1.0)/GB[5]);
    }

    // When Mc,C = Mc,C,max on AGB
    {
        double tau = tscls[1] + tscls[2];
        mc2 = mcgbtf(tau, GB[7], GB, tscls[6], tscls[7], tscls[8]);
        mcmax = std::max(std::max(mch, 0.773*mcbagb - 0.35), 1.05*mc2);
    }

    if (mcmax <= mc1) {
        if (mcmax <= GB[6]) {
            tscls[13] = tscls[6] - (1.0/((GB[4]-1.0)*GB[7]*GB[3])) *
                        std::pow(mcmax, 1.0-GB[4]);
        } else {
            tscls[13] = tscls[7] - (1.0/((GB[5]-1.0)*GB[7]*GB[2])) *
                        std::pow(mcmax, 1.0-GB[5]);
        }
    } else {
        lambda = std::min(0.9, 0.3 + 0.001*std::pow(mass,5));
        mcmax = (mcmax - lambda*mc1)/(1.0 - lambda);
        if (mcmax <= GB[6]) {
            tscls[13] = tscls[9] - (1.0/((GB[4]-1.0)*GB[1]*GB[3])) *
                        std::pow(mcmax, 1.0-GB[4]);
        } else {
            tscls[13] = tscls[10] - (1.0/((GB[5]-1.0)*GB[1]*GB[2])) *
                        std::pow(mcmax, 1.0-GB[5]);
        }
    }
    tscls[13] = std::max(tbagb, tscls[13]);
    if (mass >= 100.0) {
        tn = tscls[1];
        goto label_100;
    }

    // Nuclear timescale
    if (std::abs(mt-mcbagb) < 1.0e-14 && kw < 5) {
        tn = tbagb;
    } else {
        if (mt > mcbagb || (mt >= mc1 && kw > 4)) {
            if (kw == 6) {
                lambda = std::min(0.9, 0.3 + 0.001*std::pow(mass,5));
                mc1 = (mt - lambda*mc1)/(1.0 - lambda);
            } else {
                mc1 = mt;
            }
            if (mc1 <= GB[6]) {
                tn = tscls[9] - (1.0/((GB[4]-1.0)*GB[1]*GB[3])) *
                     std::pow(mc1, 1.0-GB[4]);
            } else {
                tn = tscls[10] - (1.0/((GB[5]-1.0)*GB[1]*GB[2])) *
                     std::pow(mc1, 1.0-GB[5]);
            }
        } else {
            if (mass > zpars_[2]) {
                mc1 = mcheif(mass, zpars_[1], zpars_[9]);
                if (mt <= mc1) {
                    tn = tscls[1];
                } else {
                    tn = tscls[1] + tscls[2]*((mt - mc1)/(mcbagb - mc1));
                }
            } else if (mass <= zpars_[1]) {
                mc1 = mcgbf(lums[2], GB, lums[5]);
                mc2 = mcgbf(lums[3], GB, lums[5]);
                if (mt <= mc1) {
                    tn = tscls[0];
                } else if (mt <= mc2) {
                    if (mt <= GB[6]) {
                        tn = tscls[3] - (1.0/((GB[4]-1.0)*GB[0]*GB[3])) *
                             std::pow(mt, 1.0-GB[4]);
                    } else {
                        tn = tscls[4] - (1.0/((GB[5]-1.0)*GB[0]*GB[2])) *
                             std::pow(mt, 1.0-GB[5]);
                    }
                } else {
                    tn = tscls[1] + tscls[2]*((mt - mc2)/(mcbagb - mc2));
                }
            } else {
                mc1 = mcheif(mass, zpars_[1], zpars_[8]);
                mc2 = mcheif(mass, zpars_[1], zpars_[9]);
                if (mt <= mc1) {
                    tn = tscls[0];
                } else if (mt <= mc2) {
                    tn = tscls[0] + tgb*((mt - mc1)/(mc2 - mc1));
                } else {
                    tn = tscls[1] + tscls[2]*((mt - mc2)/(mcbagb - mc2));
                }
            }
        }
    }
    tn = std::min(tn, tscls[13]);

    goto label_100;

label_90:
    // Helium star
    tm = themsf(mass);
    tscls[0] = tm;
    lums[0] = lzhef(mass);
    {
        double am = std::max(0.0, 0.85-0.08*mass);
        lums[1] = lums[0]*(1.0+0.45+am);
    }
    GB[7] = 8.0e-05;
    GB[2] = 4.1e+04;
    GB[3] = 5.5e+04/(1.0+0.4*std::pow(mass,4));
    GB[4] = 5.0;
    GB[5] = 3.0;
    GB[6] = std::pow(GB[2]/GB[3], 1.0/(GB[4]-GB[5]));
    lums[5] = GB[3]*std::pow(GB[6],GB[4]);

    mc1 = mcgbf(lums[1], GB, lums[5]);
    tscls[3] = tm + (1.0/((GB[4]-1.0)*GB[7]*GB[3])) *
                     std::pow(mc1, 1.0-GB[4]);
    tscls[5] = tscls[3] - (tscls[3] - tm) *
                     std::pow(GB[6]/mc1, 1.0-GB[4]);
    tscls[4] = tscls[5] + (1.0/((GB[5]-1.0)*GB[7]*GB[2])) *
                     std::pow(GB[6], 1.0-GB[5]);

    {
        double mtc = std::min(mt, 1.45*mt-0.31);
        if (mtc <= 0.0) mtc = mt;
        mcmax = std::min(mtc, std::max(mch, 0.773*mass-0.35));
    }
    if (mcmax <= GB[6]) {
        tscls[13] = tscls[3] - (1.0/((GB[4]-1.0)*GB[7]*GB[3])) *
                    std::pow(mcmax, 1.0-GB[4]);
    } else {
        tscls[13] = tscls[4] - (1.0/((GB[5]-1.0)*GB[7]*GB[2])) *
                    std::pow(mcmax, 1.0-GB[5]);
    }
    tscls[13] = std::max(tscls[13], tm);
    tn = tscls[13];

    goto label_100;

label_95:
    tm = 1.0e+10;
    tscls[0] = tm;
label_96:
    tn = 1.0e+10;

label_100:
    (void)mass0;
    return;
}

// =========================================================================
// compute_remnant — centralizes NS/BH/PPISN logic
// mc:        CO core mass at collapse
// mcbagb:    He core mass at BAGB (or ZAMS mass for He stars)
// mt_presn:  total mass just before SN
// mass0:     initial (ZAMS) mass of progenitor
// from_he_star: true if coming from naked He-star channel
// =========================================================================
RemnantResult SSEContext::compute_remnant(double mc, double mcbagb,
                                          double mt_presn, double mass0,
                                          bool from_he_star) const {
    const int nsflag = static_cast<int>(cfg_.remnant);
    RemnantResult res;
    res.fbfac = 0.0;

    // ---- ECSN: electron-capture SN for low-mass He cores ----
    if (cfg_.ecsn == ECSNModel::ON && mcbagb >= 1.6 && mcbagb <= 2.25 && mc >= 1.372) {
        res.mt = 1.26;
        res.kw = 13;  // NS
        res.fbfac = 0.0;
        return res;
    }

    // ---- PISN: pair-instability SN (complete disruption) ----
    if (cfg_.ppisn != PPISNModel::NONE && mcbagb >= 65.0 && mcbagb <= 135.0) {
        res.mt = 0.0;
        res.kw = 15;  // massless remnant
        res.fbfac = 0.0;
        return res;
    }

    // ---- PPISN: pulsational pair-instability caps BH mass ----
    double mt_cap = 1.0e30;  // effectively no cap
    if (cfg_.ppisn == PPISNModel::STRONG) {
        mt_cap = 45.0;
    } else if (cfg_.ppisn == PPISNModel::MODERATE) {
        // Leung 2019 moderate piecewise:
        //   mcbagb < 32 → no cap
        //   32-37: linear ramp 32 → cap at 40.5
        //   37-60: cap at 40.5
        if (mcbagb >= 32.0 && mcbagb < 37.0) {
            mt_cap = 32.0 + (40.5-32.0) * (mcbagb-32.0)/5.0;
        } else if (mcbagb >= 37.0 && mcbagb < 60.0) {
            mt_cap = 40.5;
        } else if (mcbagb >= 60.0 && mcbagb < 65.0) {
            // taper to PISN gap
            mt_cap = 40.5 * (65.0 - mcbagb) / 5.0;
        }
    } else if (cfg_.ppisn == PPISNModel::WEAK) {
        // Leung 2019 weak: higher cap
        if (mcbagb >= 32.0 && mcbagb < 37.0) {
            mt_cap = 32.0 + (50.0-32.0) * (mcbagb-32.0)/5.0;
        } else if (mcbagb >= 37.0 && mcbagb < 60.0) {
            mt_cap = 50.0;
        } else if (mcbagb >= 60.0 && mcbagb < 65.0) {
            mt_cap = 50.0 * (65.0 - mcbagb) / 5.0;
        }
    }

    // ---- Proto-compact-object mass (mcx) per remnant model ----
    double mcx = 0.0;
    double mt = mt_presn;

    if (nsflag == 0) {
        // Hurley 2000: linear
        mt = 1.17 + 0.09 * mc;
        res.mt = std::min(mt, mt_cap);
        res.kw = (res.mt <= cfg_.effective_mxns()) ? 13 : 14;
        res.fbfac = 0.0;
        return res;
    } else if (nsflag == 1) {
        // Belczynski 2002
        if (mc < 2.5)
            mcx = 0.161767*mc + 1.067055;
        else
            mcx = 0.314154*mc + 0.686088;
    } else if (nsflag == 2) {
        // Belczynski 2008
        if (mc < 4.82)
            mcx = 1.5;
        else if (mc < 6.31)
            mcx = 2.11;
        else if (mc < 6.75)
            mcx = 0.69315*mc - 2.26;
        else
            mcx = 0.37*mc - 0.0828;
    } else if (nsflag == 3) {
        // Fryer 2012 rapid: fixed proto-CO mass
        mcx = 1.0;
    } else if (nsflag == 4) {
        // Fryer 2012 delayed: staircase
        if (mc < 3.5)
            mcx = 1.2;
        else if (mc < 6.0)
            mcx = 1.3;
        else if (mc < 11.0)
            mcx = 1.4;
        else
            mcx = 1.6;
    } else {
        // Eldridge & Tout 2004: quadratic
        if (mc <= 2.5)
            mcx = 0.161767*mc + 1.067055;
        else
            mcx = 0.314154*mc + 0.686088;
    }

    // ---- Fallback fraction ----
    double fb = 0.0;

    if (nsflag == 1 || nsflag == 2) {
        // Belczynski: partial fallback for mc 5-7.6, full above 7.6
        if (mc <= 5.0) {
            mt = mcx;
        } else if (mc < 7.6) {
            fb = (mc - 5.0) / 2.6;
            mt = mcx + fb * (mt_presn - mcx);
        } else {
            fb = 1.0;
            mt = mt_presn;  // full fallback → direct collapse
        }
    } else if (nsflag == 3) {
        // Fryer 2012 rapid: piecewise with island of explodability
        if (mc < 2.5) {
            fb = 0.2 / (mc - 1.0);
        } else if (mc < 3.5) {
            fb = 0.286*mc - 0.514;
        } else if (mc < 11.0) {
            // island: linear ramp down then up
            double a1 = 0.25 - 1.275/(mc - 1.0);
            double b1 = -11.0*a1 + 1.0;
            fb = a1*mc + b1;
        } else {
            fb = 1.0;
        }
        fb = std::max(0.0, std::min(1.0, fb));
        mt = mcx + fb * (mt_presn - mcx);
    } else if (nsflag == 4) {
        // Fryer 2012 delayed: smooth prescription
        if (mc < 2.5) {
            fb = 0.2 / (mc - 1.0);
        } else if (mc < 3.5) {
            fb = 0.5*mc - 1.05;
        } else if (mc < 11.0) {
            double a2 = 0.133 - 0.093 / (mc - 1.0);
            double b2 = -11.0*a2 + 1.0;
            fb = a2*mc + b2;
        } else {
            fb = 1.0;
        }
        fb = std::max(0.0, std::min(1.0, fb));
        mt = mcx + fb * (mt_presn - mcx);
    } else if (nsflag == 5) {
        // Eldridge & Tout 2004: direct (no separate fallback)
        mt = mcx;
        fb = 0.0;
    }
    res.fbfac = fb;

    // ---- Gravitational mass correction (nsflag >= 1) ----
    if (nsflag >= 1) {
        // Lattimer & Yahil gravitational mass: m_grav = (-1+sqrt(1+0.3*m_bary))/0.15
        double mt_grav = (-1.0 + std::sqrt(1.0 + 0.3*mt)) / 0.15;
        if (mt_grav <= cfg_.effective_mxns()) {
            // NS: use gravitational mass
            mt = mt_grav;
        } else {
            // BH: lose 10% to neutrinos from baryonic mass
            mt = 0.9 * mt;
        }
    }

    // Apply PPISN cap
    mt = std::min(mt, mt_cap);

    // Classify
    if (mt <= 0.0) {
        res.mt = 0.0;
        res.kw = 15;
    } else if (mt <= cfg_.effective_mxns()) {
        res.mt = mt;
        res.kw = 13;  // NS
    } else {
        res.mt = mt;
        res.kw = 14;  // BH
    }

    return res;
}

// =========================================================================
// hrdiag - port of hrdiag.f
// Computes new mass, luminosity, radius & stellar type.
// =========================================================================
void SSEContext::hrdiag(double& mass, double& aj, double& mt, double& tm, double& tn,
                        double tscls[20], double lums[10], double GB[10],
                        double& r, double& lum, int& kw, double& mc, double& rc,
                        double& menv, double& renv) const {
    const double mch = 1.44, mlp = 12.0, tiny = 1.0e-14;
    const double taumin = 5.0e-08;
    const double lum0 = 7.0e+04, kap = -0.5, ahe = 4.0, aco = 16.0;

    double mass0 = mass;
    if (mass0 > 100.0) mass = 100.0;
    double mt0 = mt;
    if (mt0 > 100.0) mt = 100.0;

    if (kw > 6) goto label_90;

    {
        double tbagb = tscls[1] + tscls[2];
        double thg = tscls[0] - tm;
        double rzams = rzamsf(mass);
        double rtms = rtmsf(mass);

        if (aj < tscls[0]) {
            // On MS or HG
            double rg = rgbf(mt, lums[2]);

            if (aj < tm) {
                // Main sequence
                mc = 0.0;
                double tau = aj / tm;
                double thook = thookf(mass) * tscls[0];
                double zeta = 0.01;
                double tau1 = std::min(1.0, aj/thook);
                double tau2 = std::max(0.0,
                    std::min(1.0, (aj-(1.0-zeta)*thook)/(zeta*thook)));

                double dell = lhookf(mass, zpars_[0]);
                double dtau = tau1*tau1 - tau2*tau2;
                double alpha = lalphf(mass);
                double beta = lbetaf(mass);
                double eta = lnetaf(mass);
                double lx = std::log10(lums[1]/lums[0]);
                double xx;
                if (tau > taumin)
                    xx = alpha*tau + beta*std::pow(tau,eta) +
                         (lx - alpha - beta)*tau*tau - dell*dtau;
                else
                    xx = alpha*tau + (lx - alpha)*tau*tau - dell*dtau;
                lum = lums[0] * std::pow(10.0, xx);

                double delr = rhookf(mass, zpars_[0]);
                dtau = tau1*tau1*tau1 - tau2*tau2*tau2;
                alpha = ralphf(mass);
                beta = rbetaf(mass);
                double gamma = rgammf(mass);
                double rx = std::log10(rtms/rzams);
                if (tau > taumin)
                    xx = alpha*tau + beta*std::pow(tau,10) + gamma*std::pow(tau,40) +
                         (rx - alpha - beta - gamma)*std::pow(tau,3) - delr*dtau;
                else
                    xx = alpha*tau + (rx - alpha)*std::pow(tau,3) - delr*dtau;
                r = rzams * std::pow(10.0, xx);

                if (mass < (zpars_[0] - 0.3)) {
                    kw = 0;
                    double rdgen = 0.0258 * std::pow(1.0+zpars_[10], 5.0/3.0) *
                                   std::pow(mass, -1.0/3.0);
                    r = std::max(rdgen, r);
                } else {
                    kw = 1;
                }

            } else {
                // HG
                double mcx = mc;
                if (mass <= zpars_[1])
                    mc = mcgbf(lums[2], GB, lums[5]);
                else if (mass <= zpars_[2])
                    mc = mcheif(mass, zpars_[1], zpars_[8]);
                else
                    mc = mcheif(mass, zpars_[1], zpars_[9]);

                double eta = mctmsf(mass);
                double tau = (aj - tm) / thg;
                mc = ((1.0 - tau)*eta + tau) * mc;
                mc = std::max(mc, mcx);

                if (mc >= mt) {
                    aj = 0.0;
                    if (mass > zpars_[1]) {
                        // Zero-age helium star
                        mc = 0.0;
                        mass = mt;
                        kw = 7;
                        star(kw, mass, mt, tm, tn, tscls, lums, GB);
                    } else {
                        // Zero-age helium white dwarf
                        mc = mt;
                        mass = mt;
                        kw = 10;
                    }
                } else {
                    lum = lums[1] * std::pow(lums[2]/lums[1], tau);
                    double rx;
                    if (mass <= zpars_[2]) {
                        rx = rg;
                    } else {
                        double rmin_v = rminf(mass);
                        double ry = ragbf(mt, lums[3], zpars_[1]);
                        rx = std::min(rmin_v, ry);
                        if (mass <= mlp) {
                            double texp = std::log(mass/mlp) / std::log(zpars_[2]/mlp);
                            rx = rg;
                            rx = rmin_v * std::pow(rx/rmin_v, texp);
                        }
                        double tau2 = tblf(mass, zpars_[1], zpars_[2]);
                        if (tau2 < tiny) rx = ry;
                    }
                    r = rtms * std::pow(rx/rtms, tau);
                    kw = 2;
                }
            }

        } else if (aj < tscls[1]) {
            // Red Giant
            kw = 3;
            lum = lgbtf(aj, GB[0], GB, tscls[3], tscls[4], tscls[5]);
            if (mass <= zpars_[1]) {
                mc = mcgbf(lum, GB, lums[5]);
            } else {
                double tau = (aj - tscls[0]) / (tscls[1] - tscls[0]);
                double mcx = mcheif(mass, zpars_[1], zpars_[8]);
                double mcy = mcheif(mass, zpars_[1], zpars_[9]);
                mc = mcx + (mcy - mcx)*tau;
            }
            r = rgbf(mt, lum);
            if (mc >= mt) {
                aj = 0.0;
                if (mass > zpars_[1]) {
                    // Zero-age helium star
                    mc = 0.0;
                    mass = mt;
                    kw = 7;
                    star(kw, mass, mt, tm, tn, tscls, lums, GB);
                } else {
                    // Zero-age helium white dwarf
                    mc = mt;
                    mass = mt;
                    kw = 10;
                }
            }

        } else if (aj < tbagb) {
            // Core Helium Burning
            if (kw == 3 && mass <= zpars_[1]) {
                mass = mt;
                star(kw, mass, mt, tm, tn, tscls, lums, GB);
                aj = tscls[1];
            }
            double mcx;
            if (mass <= zpars_[1])
                mcx = mcgbf(lums[3], GB, lums[5]);
            else
                mcx = mcheif(mass, zpars_[1], zpars_[9]);
            double tau = (aj - tscls[1]) / tscls[2];
            mc = mcx + (mcagbf(mass) - mcx)*tau;

            if (mass <= zpars_[1]) {
                double lx = lums[4];
                double ly = lums[6];
                double rx = rzahbf(mt, mc, zpars_[1]);
                double rg2 = rgbf(mt, lx);
                double rmin_v = rg2 * std::pow(zpars_[12], mass/zpars_[1]);
                double texp = std::min(std::max(0.4, rmin_v/rx), 2.5);
                double ry = ragbf(mt, ly, zpars_[1]);
                double taul, tauh;
                if (rmin_v < rx) {
                    taul = std::pow(std::log(rx/rmin_v), 1.0/3.0);
                } else {
                    rmin_v = rx;
                    taul = 0.0;
                }
                tauh = std::pow(std::log(ry/rmin_v), 1.0/3.0);
                double tau2 = taul*(tau - 1.0) + tauh*tau;
                r = rmin_v * std::exp(std::abs(tau2)*std::abs(tau2)*std::abs(tau2));
                lum = lx * std::pow(ly/lx, std::pow(tau, texp));
            } else if (mass > zpars_[2]) {
                // HM stars
                double tau2 = tblf(mass, zpars_[1], zpars_[2]);
                double tloop = tscls[1] + tau2*tscls[2];
                double rmin_v = rminf(mass);
                double rg2 = rgbf(mt, lums[3]);
                double rx = ragbf(mt, lums[3], zpars_[1]);
                rmin_v = std::min(rmin_v, rx);
                if (mass <= mlp) {
                    double texp_v = std::log(mass/mlp)/std::log(zpars_[2]/mlp);
                    rx = rg2;
                    rx = rmin_v * std::pow(rx/rmin_v, texp_v);
                } else {
                    rx = rmin_v;
                }
                double texp = std::min(std::max(0.4, rmin_v/rx), 2.5);
                lum = lums[3] * std::pow(lums[6]/lums[3], std::pow(tau, texp));
                if (aj < tloop) {
                    double ly = lums[3] * std::pow(lums[6]/lums[3], std::pow(tau2, texp));
                    double ry = ragbf(mt, ly, zpars_[1]);
                    double taul = 0.0;
                    if (std::abs(rmin_v-rx) > tiny)
                        taul = std::pow(std::log(rx/rmin_v), 1.0/3.0);
                    double tauh = 0.0;
                    if (ry > rmin_v) tauh = std::pow(std::log(ry/rmin_v), 1.0/3.0);
                    double tau_v = (aj - tscls[1])/(tau2*tscls[2]);
                    double tau2v = taul*(tau_v - 1.0) + tauh*tau_v;
                    r = rmin_v * std::exp(std::abs(tau2v)*std::abs(tau2v)*std::abs(tau2v));
                } else {
                    r = ragbf(mt, lum, zpars_[1]);
                }
            } else {
                // IM stars
                double tau2 = 1.0 - tblf(mass, zpars_[1], zpars_[2]);
                double tloop = tscls[1] + tau2*tscls[2];
                if (aj < tloop) {
                    double tau_v = (tloop - aj)/(tau2*tscls[2]);
                    lum = lums[4] * std::pow(lums[3]/lums[4], std::pow(tau_v,3));
                    r = rgbf(mt, lum);
                } else {
                    double lx = lums[4];
                    double ly = lums[6];
                    double rx2 = rgbf(mt, lx);
                    double rmin_v = rminf(mt);
                    double texp = std::min(std::max(0.4, rmin_v/rx2), 2.5);
                    double ry = ragbf(mt, ly, zpars_[1]);
                    double taul, tauh;
                    if (rmin_v < rx2) {
                        taul = std::pow(std::log(rx2/rmin_v), 1.0/3.0);
                    } else {
                        rmin_v = rx2;
                        taul = 0.0;
                    }
                    tauh = std::pow(std::log(ry/rmin_v), 1.0/3.0);
                    double tau_v = (aj - tloop)/(tscls[2] - (tloop - tscls[1]));
                    double tau2v = taul*(tau_v - 1.0) + tauh*tau_v;
                    r = rmin_v * std::exp(std::abs(tau2v)*std::abs(tau2v)*std::abs(tau2v));
                    lum = lx * std::pow(ly/lx, std::pow(tau_v, texp));
                }
            }

            // Test whether core mass exceeds total mass
            if (mc >= mt) {
                // Evolved MS naked helium star
                kw = 7;
                double xx = (aj - tscls[1])/tscls[2];
                mass = mt;
                star(kw, mass, mt, tm, tn, tscls, lums, GB);
                aj = xx*tm;
            } else {
                kw = 4;
            }

        } else {
            // AGB
            double mcbagb = mcagbf(mass);
            double mcx = mcgbtf(tbagb, GB[7], GB, tscls[6], tscls[7], tscls[8]);
            double mcmax = std::max(std::max(mch, 0.773*mcbagb - 0.35), 1.05*mcx);

            if (aj < tscls[12]) {
                mcx = mcgbtf(aj, GB[7], GB, tscls[6], tscls[7], tscls[8]);
                mc = mcbagb;
                lum = lmcgbf(mcx, GB);
                if (mt <= mc) {
                    // Evolved naked helium star (post-HeMS)
                    kw = 9;
                    mt = mc;
                    mass = mt;
                    mc = mcx;
                    star(kw, mass, mt, tm, tn, tscls, lums, GB);
                    if (mc <= GB[6])
                        aj = tscls[3] - (1.0/((GB[4]-1.0)*GB[7]*GB[3])) *
                             std::pow(mc, 1.0-GB[4]);
                    else
                        aj = tscls[4] - (1.0/((GB[5]-1.0)*GB[7]*GB[2])) *
                             std::pow(mc, 1.0-GB[5]);
                    aj = std::max(aj, tm);
                    goto label_90;
                } else {
                    kw = 5;
                }
            } else {
                kw = 6;
                mc = mcgbtf(aj, GB[1], GB, tscls[9], tscls[10], tscls[11]);
                lum = lmcgbf(mc, GB);
                // 3rd dredge-up
                double lambda = std::min(0.9, 0.3 + 0.001*std::pow(mass,5));
                double tau_v = tscls[12];
                mcx = mcgbtf(tau_v, GB[1], GB, tscls[9], tscls[10], tscls[11]);
                double mcy = mc;
                mc = mc - lambda*(mcy - mcx);
                mcx = mc;
                mcmax = std::min(mt, mcmax);
            }
            r = ragbf(mt, lum, zpars_[1]);

            // Check if core mass exceeds mcmax
            if (mcmax - mcx < tiny) {
                aj = 0.0;
                mc = mcmax;
                if (mc < mch) {
                    // WD IFMR
                    if (cfg_.ifflag >= 1) {
                        if (zpars_[13] >= 1.0e-08) {
                            mc = std::min(0.36+0.104*mass, 0.58+0.061*mass);
                            mc = std::max(0.54+0.042*mass, mc);
                            if (mass < 1.0) mc = 0.46;
                        } else {
                            mc = std::min(0.29+0.178*mass, 0.65+0.062*mass);
                            mc = std::max(0.54+0.073*mass, mc);
                        }
                        mc = std::min(mch, mc);
                    }
                    mt = mc;
                    mass = mt;
                    if (mcbagb < 1.6)
                        kw = 11; // CO WD
                    else
                        kw = 12; // ONe WD
                } else {
                    if (mcbagb < 1.6) {
                        kw = 15; // massless SN
                        mt = 0.0;
                        lum = 1.0e-10;
                        r = 1.0e-10;
                    } else {
                        auto rem = compute_remnant(mc, mcbagb, mt, mass, false);
                        mt = rem.mt;
                        kw = rem.kw;
                        mc = mt;
                        if (kw == 15) { lum = 1.0e-10; r = 1.0e-10; }
                    }
                }
            }
        }
    }

label_90:
    if (kw >= 7 && kw <= 9) {
        // Naked Helium Star
        double rzams_he = rzhef(mt);
        double rx = rzams_he;
        if (aj < tm) {
            kw = 7;
            double tau = aj/tm;
            double am = std::max(0.0, 0.85-0.08*mass);
            lum = lums[0]*(1.0+0.45*tau+am*tau*tau);
            am = std::max(0.0, 0.4-0.22*std::log10(mt));
            r = rx*(1.0+am*(tau-std::pow(tau,6)));
            mc = 0.0;
            if (mt < zpars_[9]) kw = 10;
        } else {
            kw = 8;
            lum = lgbtf(aj, GB[7], GB, tscls[3], tscls[4], tscls[5]);
            r = rhehgf(mt, lum, rx, lums[1]);
            double rg = rhegbf(lum);
            if (r >= rg) { kw = 9; r = rg; }
            mc = mcgbf(lum, GB, lums[5]);
            double mtc = std::min(mt, 1.45*mt-0.31);
            double mcmax = std::min(mtc, std::max(mch, 0.773*mass-0.35));
            if (mcmax - mc < tiny) {
                aj = 0.0;
                mc = mcmax;
                if (mc < mch) {
                    if (mass < 1.6) {
                        mt = std::max(mc, (mc+0.31)/1.45);
                        kw = 11;
                    } else {
                        mt = mc;
                        kw = 12;
                    }
                    mass = mt;
                } else {
                    if (mass < 1.6) {
                        kw = 15;
                        mt = 0.0;
                        lum = 1.0e-10;
                        r = 1.0e-10;
                    } else {
                        auto rem = compute_remnant(mc, mass, mt, mass, true);
                        mt = rem.mt;
                        kw = rem.kw;
                        mc = mt;
                        if (kw == 15) { lum = 1.0e-10; r = 1.0e-10; }
                    }
                }
            }
        }
    }

    if (kw >= 10 && kw <= 12) {
        // White Dwarf
        mc = mt;
        if (mc >= mch) {
            if (kw == 12) { kw = 13; aj = 0.0; mt = 1.3; }
            else { kw = 15; aj = 0.0; mt = 0.0; lum = 1.0e-10; r = 1.0e-10; }
        } else {
            double xx = (kw == 10) ? ahe : aco;
            if (cfg_.wdflag == 0) {
                lum = 635.0*mt*zpars_[13] / std::pow(xx*(aj+0.1), 1.4);
            } else {
                if (aj < 9000.0) {
                    lum = 300.0*mt*zpars_[13] / std::pow(xx*(aj+0.1), 1.18);
                } else {
                    double fac = std::pow(9000.1*xx, 5.3);
                    lum = 300.0*fac*mt*zpars_[13] / std::pow(xx*(aj+0.1), 6.48);
                }
            }
            r = 0.0115 * std::sqrt(std::max(1.48204e-06,
                std::pow(mch/mt, 2.0/3.0) - std::pow(mt/mch, 2.0/3.0)));
            r = std::min(0.1, r);
            if (mt < 0.0005) r = 0.09;
            if (mt < 0.000005) r = 0.009;
        }
    }

    if (kw == 13) {
        mc = mt;
        if (mc > cfg_.effective_mxns()) {
            kw = 14;
            aj = 0.0;
        } else {
            lum = 0.02 * std::pow(mt, 0.67) / std::pow(std::max(aj, 0.1), 2);
            r = 1.4e-05;
        }
    }

    if (kw == 14) {
        mc = mt;
        lum = 1.0e-10;
        r = 4.24e-06 * mt;
    }

    // Core radius
    rc = 0.0;
    if (kw <= 1 || kw == 7) {
        rc = 0.0;
    } else if (kw <= 3) {
        if (mass > zpars_[1]) {
            rc = rzhef(mc);
        } else {
            double rx_v = 0.0115*std::sqrt(std::max(1.48204e-06,
                   std::pow(mch/mc, 2.0/3.0) - std::pow(mc/mch, 2.0/3.0)));
            rc = 5.0*rx_v;
        }
    } else if (kw == 4) {
        // CHeB core radius: He-ZAMS radius at mc
        rc = rzhef(mc);
    } else if (kw >= 10) {
        rc = r;
    }
    menv = 0.0;
    renv = 0.0;

    // Perturbation for small envelope mass
    if (kw >= 2 && kw <= 9 && kw != 7) {
        double mtc_v = mt;
        if (kw >= 8) mtc_v = std::min(mt, 1.45*mt-0.31);
        double mew = ((mt-mc)/mt) * std::min(5.0, std::max(1.2, std::pow(lum/lum0, kap)));
        if (kw >= 8) mew = ((mtc_v-mc)/mtc_v) * 5.0;
        if (mew < 1.0) {
            double lx_v, rx_v;
            if (kw <= 3) {
                if (mass > zpars_[1]) {
                    lx_v = lzhef(mc);
                    rx_v = rzhef(mc);
                } else {
                    if (cfg_.wdflag == 0)
                        lx_v = 635.0*mc*zpars_[13] / std::pow(ahe*0.1, 1.4);
                    else
                        lx_v = 300.0*mc*zpars_[13] / std::pow(ahe*0.1, 1.18);
                    rx_v = 0.0115*std::sqrt(std::max(1.48204e-06,
                           std::pow(mch/mc, 2.0/3.0) - std::pow(mc/mch, 2.0/3.0)));
                    rx_v = 5.0*rx_v;
                }
            } else if (kw <= 6) {
                if (cfg_.wdflag == 0)
                    lx_v = 635.0*mc*zpars_[13] / std::pow(aco*0.1, 1.4);
                else
                    lx_v = 300.0*mc*zpars_[13] / std::pow(aco*0.1, 1.18);
                rx_v = 0.0115*std::sqrt(std::max(1.48204e-06,
                       std::pow(mch/mc, 2.0/3.0) - std::pow(mc/mch, 2.0/3.0)));
                rx_v = 5.0*rx_v;
            } else {
                // kw 8,9
                if (cfg_.wdflag == 0)
                    lx_v = 635.0*mc*zpars_[13] / std::pow(aco*0.1, 1.4);
                else
                    lx_v = 300.0*mc*zpars_[13] / std::pow(aco*0.1, 1.18);
                rx_v = 0.0115*std::sqrt(std::max(1.48204e-06,
                       std::pow(mch/mc, 2.0/3.0) - std::pow(mc/mch, 2.0/3.0)));
                rx_v = 5.0*rx_v;
            }
            double xx = lpertf(mt, mew);
            lum = lx_v * std::pow(lum/lx_v, xx);
            if (r <= rx_v)
                xx = 0.0;
            else
                xx = rpertf(mt, mew, r, rx_v);
            r = rx_v * std::pow(r/rx_v, xx);
        }
    }

    if (mass > 99.99) mass = mass0;
    if (mt > 99.99) mt = mt0;
}

// =========================================================================
// evolve - port of evolv1.f
// Main evolution loop with mass loss integration.
// =========================================================================
SSEResult SSEContext::evolve(double m_zams, double age_myr) const {
    const double tiny = 1.0e-14;
    const double tol = 1.0e-10;
    const double eps = 1.0e-06;
    const double alpha2 = 0.09;
    const int nv = 50000;

    int kw = 1;
    double mass = m_zams;
    double mt = mass;
    double epoch = 0.0;
    double tphys = 0.0;
    double tphysf = age_myr;

    double tm = 0.0, tn = 0.0;
    double tscls[20], lums[10], GB[10];
    std::fill(tscls, tscls + 20, 0.0);
    std::fill(lums, lums + 10, 0.0);
    std::fill(GB, GB + 10, 0.0);

    double r = 0.0, lum = 0.0, mc = 0.0, rc = 0.0;
    double mc1 = 0.0, menv = 0.0, renv = 0.0;
    double rl = 0.0;
    double dtm = 0.0;
    double rm0 = 0.0;
    double ajhold = 0.0;

    for (int j = 1; j <= nv; j++) {
        double dms = 0.0;

        if (cfg_.neta > tiny && j > 1) {
            double dt = 1.0e+06 * dtm;
            dms = mlwind(kw, lum, r, mt, mc, rl, Z_) * dt;
            if (kw < 10) {
                double dml = mt - mc;
                if (dml < dms) {
                    dtm = (dml/dms) * dtm;
                    dms = dml;
                }
            }
        }

        // Limit to 1% mass loss
        if (dms > 0.01*mt) {
            dtm = 0.01*mt*dtm/dms;
            dms = 0.01*mt;
        }

        // Update mass and time, reset epoch for MS/HG star
        // Note: m0, tmold, tbgold set inside dms>0 block to match Fortran
        double m0 = mass;
        double tmold = tm;
        double tbgold = tscls[0];

        if (dms > 0.0) {
            mt = mt - dms;
            if (kw <= 2 || kw == 7) {
                m0 = mass;
                mc1 = mc;
                mass = mt;
                tmold = tm;
                tbgold = tscls[0];
                star(kw, mass, mt, tm, tn, tscls, lums, GB);
                if (kw == 2) {
                    if (GB[8] < mc1 || m0 > zpars_[2]) {
                        mass = m0;
                    } else {
                        epoch = tm + (tscls[0] - tm)*(ajhold-tmold)/(tbgold - tmold);
                        epoch = tphys - epoch;
                    }
                } else {
                    epoch = tphys - ajhold*tm/tmold;
                }
            }
        }

        tphys = tphys + dtm;

        double aj = tphys - epoch;
        star(kw, mass, mt, tm, tn, tscls, lums, GB);

        int kwold = kw;
        hrdiag(mass, aj, mt, tm, tn, tscls, lums, GB,
               r, lum, kw, mc, rc, menv, renv);

        // Check for type change / end condition
        if (kw == 15) {
            return {15, 0.0, 0.0, 0.0, 0.0};
        }

        if (tphys >= tphysf) {
            return {kw, mt, mc, r, lum};
        }

        // Record radius and age
        rm0 = r;
        ajhold = aj;
        if (kw != kwold) kwold = kw;

        epoch = tphys - aj;
        deltat(kw, aj, tm, tn, tscls, dtm, dtm);

        // Check for type change approaching
        double dtr;
        deltat(kw, aj, tm, tn, tscls, dtm, dtr);

        m0 = mass;
        if ((dtr - dtm) <= tol && kw <= 9) {
            // Near type change - check final radius using copies
            double aj_c = std::max(aj, aj*(1.0-eps)+dtr);
            double mc1_c = mc, mt_c = mt, tm_c = tm, tn_c = tn, mass_c = mass;
            double r1, lum1, rc1, menv1, renv1;
            double tscls_c[20], lums_c[10], GB_c[10];
            std::copy(tscls, tscls+20, tscls_c);
            std::copy(lums, lums+10, lums_c);
            std::copy(GB, GB+10, GB_c);
            int kw_c = kw;
            hrdiag(mass_c, aj_c, mt_c, tm_c, tn_c, tscls_c, lums_c, GB_c,
                   r1, lum1, kw_c, mc1_c, rc1, menv1, renv1);
            double dr = r1 - rm0;
            if (std::abs(dr) > 0.1*rm0) {
                dtm = dtr - ajhold*eps;
                double dtdr = dtm/std::abs(dr);
                dtm = alpha2*std::max(r1,rm0)*dtdr;
            } else {
                dtm = dtr;
                goto label_50;
            }
        }

        // Limit radius change to 10%
        {
            double aj_c = ajhold + dtm;
            double mc1_c = mc, mt_c = mt, tm_c = tm, tn_c = tn, mass_c = mass;
            double r1, lum1, rc1, menv1, renv1;
            double tscls_c[20], lums_c[10], GB_c[10];
            std::copy(tscls, tscls+20, tscls_c);
            std::copy(lums, lums+10, lums_c);
            std::copy(GB, GB+10, GB_c);
            int kw_c = kw;
            hrdiag(mass_c, aj_c, mt_c, tm_c, tn_c, tscls_c, lums_c, GB_c,
                   r1, lum1, kw_c, mc1_c, rc1, menv1, renv1);
            double dr = r1 - rm0;
            int it = 0;
            while (std::abs(dr) > 0.1*rm0) {
                it++;
                if (it == 20 && kw == 4) break;
                if (it > 30) break;
                double dtdr = dtm/std::abs(dr);
                dtm = alpha2*std::max(rm0,r1)*dtdr;
                if (it >= 20) dtm = 0.5*dtm;
                aj_c = ajhold + dtm;
                mc1_c = mc; mt_c = mt; tm_c = tm; tn_c = tn; mass_c = mass;
                kw_c = kw;
                std::copy(tscls, tscls+20, tscls_c);
                std::copy(lums, lums+10, lums_c);
                std::copy(GB, GB+10, GB_c);
                hrdiag(mass_c, aj_c, mt_c, tm_c, tn_c, tscls_c, lums_c, GB_c,
                       r1, lum1, kw_c, mc1_c, rc1, menv1, renv1);
                dr = r1 - rm0;
            }
        }

    label_50:
        // Ensure type change hasn't occurred during check
        if (kw != kwold) {
            kw = kwold;
            mass = m0;
            star(kw, mass, mt, tm, tn, tscls, lums, GB);
        }

        dtm = std::max(dtm, 1.0e-07*aj);
        dtm = std::min(dtm, tphysf - tphys);
    }

    // If we get here, max iterations reached
    return {kw, mt, mc, r, lum};
}
