# Time Series COVID-19 Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.var_model import VAR

import lmfit
from scipy.integrate import odeint

def forecast(time_series, beds, agegroups, forecast_days):
    # parameters
    gamma = 1.0 / 9.0
    sigma = 1.0 / 3.0

    outbreak_shift = 30  # let's try 30, 1, 20?

    if outbreak_shift >= 0:
        y_data = np.concatenate((np.zeros(outbreak_shift), time_series))
    else:
        y_data = time_series[-outbreak_shift:]

    days = outbreak_shift + len(time_series)
    x_data = np.linspace(0, days - 1, days, dtype=int)

    # form: {parameter: (initial guess, minimum value, max value)}
    params_init_min_max = {'R_0_start': (3.0, 2.0, 5.0),
                           'k': (2.5, 0.01, 5.0),
                           'x0': (90, 0, 120),
                           'R_0_end': (0.9, 0.3, 3.5),
                           'prob_I_to_C': (0.05, 0.01, 0.1),
                           'prob_C_to_D': (0.5, 0.05, 0.8),
                           's': (0.003, 0.001, 0.01)}

    # rate of change
    def deriv(y, t, beta, gamma, sigma, N, p_I_to_C, p_C_to_D, Beds):
        S, E, I, C, R, D = y

        dSdt = -beta(t) * I * S / N
        dEdt = beta(t) * I * S / N - sigma * E
        dIdt = sigma * E - 1 / 12.0 * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
        dCdt = 1 / 12.0 * p_I_to_C * I - 1 / 7.5 * p_C_to_D * min(Beds(t), C) - max(0, C - Beds(t)) - (1 - p_C_to_D) * \
               1 / 6.5 * min(Beds(t), C)
        dRdt = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * 1 / 6.5 * min(Beds(t), C)
        dDdt = 1 / 7.5 * p_C_to_D * min(Beds(t), C) + max(0, C - Beds(t))

        return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt

    def logistic_R_0(t, R_0_start, k, x0, R_0_end):
        return (R_0_start - R_0_end) / (1 + np.exp(-k * (-t + x0))) + R_0_end

    # SIER-CD Model
    def Model(days, agegroups, beds_per_100k, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s):

        def beta(t):
            return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma

        N = sum(agegroups)

        def Beds(t):
            beds_0 = beds_per_100k / 100_000 * N
            return beds_0 + s * beds_0 * t  # 0.003

        y0 = N - 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
        t = np.linspace(0, days, days)
        ret = odeint(deriv, y0, t, args=(beta, gamma, sigma, N, prob_I_to_C, prob_C_to_D, Beds))
        S, E, I, C, R, D = ret.T
        R_0_over_time = [beta(i) / gamma for i in range(len(t))]

        return t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D

    def fitter(x, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s):
        ret = Model(days, agegroups, beds, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s)

        return ret[6][x]

    mod = lmfit.Model(fitter)
    for kwarg, (init, mini, maxi) in params_init_min_max.items():
        mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)

    params = mod.make_params()
    result = mod.fit(y_data, params, method="least_squares", x=x_data)
    forecast_values = Model(int(y_data.shape[0] + forecast_days), agegroups, beds, **result.best_values)[6]

    return forecast_values[-forecast_days:]

def scorer(y_pred, y_test):
    return np.sum([(np.log10(np.ceil(x)) - np.log10(y)) ** 2 for x, y in zip(y_pred, y_test) if x > 1 and y > 1])

def scorer_100(y_pred, y_test):
    return np.sum([(np.log10(np.ceil(x)) - np.log10(y)) ** 2 for x, y in zip(y_pred, y_test) if x > 1 and y > 100])

if __name__ == '__main__':
    dir_path = '../COVID-19/csse_covid_19_data/csse_covid_19_time_series/'

    cases = pd.read_csv(dir_path + '/time_series_covid19_confirmed_global.csv')
    cases.fillna('None', inplace=True)

    deaths = pd.read_csv(dir_path + '/time_series_covid19_deaths_global.csv')
    deaths.fillna('None', inplace=True)

    recovered = pd.read_csv(dir_path + '/time_series_covid19_recovered_global.csv')
    recovered.fillna('None', inplace=True)

    beds = pd.read_csv('beds.csv', header=0)
    beds_lookup = dict(zip(beds["Country"], beds["ICU_Beds"]))

    agegroups = pd.read_csv('agegroups.csv')
    agegroups_lookup = dict(zip(agegroups['Location'],
                                agegroups[['0_9', '10_19', '20_29', '30_39', '40_49',
                                           '50_59', '60_69', '70_79', '80_89', '90_100']].values))

    # time series for cases, deaths, recovered
    cases['region'] = list(zip(cases['Country/Region'], cases['Province/State']))
    deaths['region'] = list(zip(deaths['Country/Region'], deaths['Province/State']))
    recovered['region'] = list(zip(recovered['Country/Region'], recovered['Province/State']))

    drop_cols = ['Country/Region', 'Province/State', 'Lat', 'Long', '1/22/20', '1/23/20', '1/24/20',
                 '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20', '1/30/20', '1/31/20']
    cases.drop(drop_cols, axis=1, inplace=True)
    deaths.drop(drop_cols, axis=1, inplace=True)
    recovered.drop(drop_cols, axis=1, inplace=True)

    # forecasting for cases, deaths, recovered
    # forecasted_values = []
    # for x in range(len(cases)):
    #     ts = np.asarray(cases.iloc[x, :-1]).astype('float64')
    #     if cases.iloc[x, -1][0] not in beds_lookup.keys():
    #         beds = 4.0      # approximate default value
    #     else:
    #         beds = beds_lookup[cases.iloc[x, -1][0]]
    #
    #     if cases.iloc[x, -1][0] not in agegroups_lookup.keys():
    #         # fallback option - 1M as total population of a country
    #         agegroups = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]
    #     else:
    #         agegroups = agegroups_lookup[cases.iloc[x, -1][0]]
    #
    #     y_pred = forecast(ts, beds, agegroups, forecast_days=3)
    #     forecasted_values.append(y_pred)
    #     print('Forecasts Complete:',x)

    # print(forecasted_values)
    # pd.DataFrame(forecasted_values, columns=['day1', 'day2', 'day3']).to_csv('cases_forecast_3days.csv')

    cases_forecast = pd.concat([cases, pd.read_csv('cases_forecast_3days.csv', index_col=0)], axis=1)

    cases_latest = pd.read_csv('time_series_covid19_confirmed_global_latest.csv')
    cases_latest.fillna('None', inplace=True)

    print('Score for Apr 21 Cases Forecast: ',scorer(list(cases_forecast.day1), list(cases_latest.iloc[:,-2])))
    print('Score for Apr 22 Cases Forecast: ',scorer(list(cases_forecast.day2), list(cases_latest.iloc[:,-1])))

    print('\nScore for Apr 21 Cases Forecast (>100 Condition): ',scorer_100(list(cases_forecast.day1), list(cases_latest.iloc[:,-2])))
    print('Score for Apr 22 Cases Forecast (>100 Condition): ',scorer_100(list(cases_forecast.day2), list(cases_latest.iloc[:,-1])))

    # forecasted_values = []
    # for x in range(len(deaths)):
    #     ts = np.asarray(deaths.iloc[x, :-1]).astype('float64')
    #     if deaths.iloc[x, -1][0] not in beds_lookup.keys():
    #         beds = 4.0      # approximate default value
    #     else:
    #         beds = beds_lookup[deaths.iloc[x, -1][0]]
    #
    #     if deaths.iloc[x, -1][0] not in agegroups_lookup.keys():
    #         # fallback option - 1M as total population of a country
    #         agegroups = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]
    #     else:
    #         agegroups = agegroups_lookup[deaths.iloc[x, -1][0]]
    #
    #     y_pred = forecast(ts, beds, agegroups, forecast_days=3)
    #     forecasted_values.append(y_pred)
    #     print('Forecasts Complete:',x)
    #
    # print(forecasted_values)
    # pd.DataFrame(forecasted_values, columns=['day1', 'day2', 'day3']).to_csv('deaths_forecast_3days.csv')

    deaths_forecast = pd.concat([cases, pd.read_csv('deaths_forecast_3days.csv', index_col=0)], axis=1)

    deaths_latest = pd.read_csv('time_series_covid19_deaths_global_latest.csv')
    deaths_latest.fillna('None', inplace=True)

    print('\nScore for Apr 21 Deaths Forecast: ', scorer(list(deaths_forecast.day1), list(deaths_latest.iloc[:, -2])))
    print('Score for Apr 22 Deaths Forecast: ', scorer(list(deaths_forecast.day2), list(deaths_latest.iloc[:, -1])))

    print('\nScore for Apr 21 Deaths Forecast (>100 Condition): ', scorer_100(list(deaths_forecast.day1), list(deaths_latest.iloc[:, -2])))
    print('Score for Apr 22 Deaths Forecast (>100 Condition): ', scorer_100(list(deaths_forecast.day2), list(deaths_latest.iloc[:, -1])))

    #
    # # covid_italy_cases = np.asarray(list(zip(*covid_italy))[0])
    # # beds = pd.read_csv('beds.csv', header=0)
    # # beds_lookup = dict(zip(beds["Country"], beds["ICU_Beds"]))
    # # beds_italy = beds_lookup['Italy']
    # #
    # # agegroups = pd.read_csv('agegroups.csv')
    # # agegroup_lookup = dict(zip(agegroups['Location'],
    # #                            agegroups[['0_9', '10_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80_89',
    # #                                       '90_100']].values))
    # # agegroups_italy = agegroups_lookup['Italy']
    #
    # days = list(cases[cases.region == ('US', 'None')])[:-1]
    # cases_us = cases[cases.region == ('US', 'None')]
    # deaths_us = deaths[deaths.region == ('US', 'None')]
    # recovered_us = recovered[recovered.region == ('US', 'None')]
    #
    # covid_us = []
    # for i in days:
    #     covid_us.append([int(cases_us[i]), int(deaths_us[i]), int(recovered_us[i])])
    #
    # covid_us_cases = list(list(zip(*covid_us))[0])
    # covid_us_deaths = list(list(zip(*covid_us))[1])
    # covid_us_recovered = list(list(zip(*covid_us))[2])

    # SIR Model for Deaths Forecast

    # # Italy!
    # cases_italy = cases[cases.region == ('Italy', 'None')]
    # deaths_italy = deaths[deaths.region == ('Italy', 'None')]
    # recovered_italy = recovered[recovered.region == ('Italy', 'None')]
    #
    # covid_italy = []
    # for i in days:
    #     covid_italy.append([int(cases_italy[i]), int(deaths_italy[i]), int(recovered_italy[i])])
    #
    # # data, parameters, etc.
    # covid_italy_deaths = np.asarray(list(zip(*covid_italy))[1])
    # beds = pd.read_csv('beds.csv', header=0)
    # beds_lookup = dict(zip(beds["Country"], beds["ICU_Beds"]))
    # beds_italy = beds_lookup['Italy']
    #
    # agegroups = pd.read_csv('agegroups.csv')
    # agegroup_lookup = dict(zip(agegroups['Location'],
    #                            agegroups[['0_9', '10_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80_89', '90_100']].values))
    # agegroups_italy = agegroup_lookup['Italy']
    #
    # probabilities = pd.read_csv('probabilities.csv')
    # prob_I_to_C_1 = list(probabilities.prob_I_to_ICU_1.values)
    # prob_I_to_C_2 = list(probabilities.prob_I_to_ICU_2.values)
    # prob_C_to_Death_1 = list(probabilities.prob_ICU_to_Death_1.values)
    # prob_C_to_Death_2 = list(probabilities.prob_ICU_to_Death_2.values)
    #
    # gamma = 1.0 / 9.0
    # sigma = 1.0 / 3.0
    #
    # # outbreak_shift = np.where(covid_italy_deaths > 0)[0][0]
    # outbreak_shift = 20 # maybe let's try 30, 1, 20?
    #
    # if outbreak_shift >= 0:
    #     y_data = np.concatenate((np.zeros(outbreak_shift), covid_italy_deaths))
    # else:
    #     y_data = covid_italy_deaths[-outbreak_shift:]
    #
    # days = outbreak_shift + len(covid_italy_deaths)
    # x_data = np.linspace(0, days - 1, days, dtype=int)
    #
    # # form: {parameter: (initial guess, minimum value, max value)}
    # params_init_min_max = {'R_0_start': (3.0, 2.0, 5.0),
    #                        'k': (2.5, 0.01, 5.0),
    #                        'x0': (90, 0, 120),
    #                        'R_0_end': (0.9, 0.3, 3.5),
    #                        'prob_I_to_C': (0.05, 0.01, 0.1),
    #                        'prob_C_to_D': (0.5, 0.05, 0.8),
    #                        's': (0.003, 0.001, 0.01)}
    #
    # # rate of change
    # def deriv(y, t, beta, gamma, sigma, N, p_I_to_C, p_C_to_D, Beds):
    #     S, E, I, C, R, D = y
    #
    #     dSdt = -beta(t) * I * S / N
    #     dEdt = beta(t) * I * S / N - sigma * E
    #     dIdt = sigma * E - 1 / 12.0 * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
    #     dCdt = 1 / 12.0 * p_I_to_C * I - 1 / 7.5 * p_C_to_D * min(Beds(t), C) - max(0, C - Beds(t)) - (1 - p_C_to_D) * \
    #            1 / 6.5 * min(Beds(t), C)
    #     dRdt = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * 1 / 6.5 * min(Beds(t), C)
    #     dDdt = 1 / 7.5 * p_C_to_D * min(Beds(t), C) + max(0, C - Beds(t))
    #
    #     return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt
    #
    # def logistic_R_0(t, R_0_start, k, x0, R_0_end):
    #     return (R_0_start - R_0_end) / (1 + np.exp(-k * (-t + x0))) + R_0_end
    #
    # # SIER-CD Model
    # def Model(days, agegroups, beds_per_100k, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s):
    #
    #     def beta(t):
    #         return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma
    #
    #     N = sum(agegroups)
    #
    #     def Beds(t):
    #         beds_0 = beds_per_100k / 100_000 * N
    #         return beds_0 + s * beds_0 * t  # 0.003
    #
    #     y0 = N - 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    #     t = np.linspace(0, days, days)
    #     ret = odeint(deriv, y0, t, args=(beta, gamma, sigma, N, prob_I_to_C, prob_C_to_D, Beds))
    #     S, E, I, C, R, D = ret.T
    #     R_0_over_time = [beta(i) / gamma for i in range(len(t))]
    #
    #     return t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D
    #
    # def fitter(x, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s):
    #     ret = Model(days, agegroups_italy, beds_italy, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s)
    #
    #     return ret[6][x]
    #
    # # fitting model
    # mod = lmfit.Model(fitter)
    #
    # for kwarg, (init, mini, maxi) in params_init_min_max.items():
    #     mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)
    #
    # params = mod.make_params()
    # fit_method = "leastsq"
    #
    # result = mod.fit(y_data, params, method="least_squares", x=x_data)
    #
    # np.set_printoptions(suppress=True)
    # # order of output: t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D
    # print(Model(int(y_data.shape[0]+3), agegroups_italy, beds_italy, **result.best_values)[6])
    # print(y_data)
    #
    # # result.plot_fit(datafmt="-", ax=1, show_init=True)
    # # plt.show()

    # # USA!
    # cases_us = cases[cases.region == ('US', 'None')]
    # deaths_us = deaths[deaths.region == ('US', 'None')]
    # recovered_us = recovered[recovered.region == ('US', 'None')]
    #
    # covid_us = []
    # for i in days:
    #     covid_us.append([int(cases_us[i]), int(deaths_us[i]), int(recovered_us[i])])
    #
    # # data, parameters, etc.
    # covid_us_deaths = np.asarray(list(zip(*covid_us))[1])
    # beds = pd.read_csv('beds.csv', header=0)
    # beds_lookup = dict(zip(beds["Country"], beds["ICU_Beds"]))
    # beds_us = beds_lookup['United States']
    #
    # agegroups = pd.read_csv('agegroups.csv')
    # agegroup_lookup = dict(zip(agegroups['Location'],
    #                            agegroups[['0_9', '10_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80_89',
    #                                       '90_100']].values))
    # agegroups_us = agegroup_lookup['US']
    #
    # probabilities = pd.read_csv('probabilities.csv')
    # prob_I_to_C_1 = list(probabilities.prob_I_to_ICU_1.values)
    # prob_I_to_C_2 = list(probabilities.prob_I_to_ICU_2.values)
    # prob_C_to_Death_1 = list(probabilities.prob_ICU_to_Death_1.values)
    # prob_C_to_Death_2 = list(probabilities.prob_ICU_to_Death_2.values)
    #
    # gamma = 1.0 / 9.0
    # sigma = 1.0 / 3.0
    #
    # # outbreak_shift = np.where(covid_us_deaths > 0)[0][0]
    # outbreak_shift = 30 # maybe let's try 30, 1, 20?
    #
    # print(covid_us_deaths)
    # if outbreak_shift >= 0:
    #     y_data = np.concatenate((np.zeros(outbreak_shift), covid_us_deaths))
    # else:
    #     y_data = covid_us_deaths[-outbreak_shift:]
    #
    # days = outbreak_shift + len(covid_us_deaths)
    # x_data = np.linspace(0, days - 1, days, dtype=int)
    #
    # # form: {parameter: (initial guess, minimum value, max value)}
    # params_init_min_max = {'R_0_start': (3.0, 2.0, 5.0),
    #                        'k': (2.5, 0.01, 5.0),
    #                        'x0': (90, 0, 120),
    #                        'R_0_end': (0.9, 0.3, 3.5),
    #                        'prob_I_to_C': (0.05, 0.01, 0.1),
    #                        'prob_C_to_D': (0.5, 0.05, 0.8),
    #                        's': (0.003, 0.001, 0.01)}
    #
    #
    # # rate of change
    # def deriv(y, t, beta, gamma, sigma, N, p_I_to_C, p_C_to_D, Beds):
    #     S, E, I, C, R, D = y
    #
    #     dSdt = -beta(t) * I * S / N
    #     dEdt = beta(t) * I * S / N - sigma * E
    #     dIdt = sigma * E - 1 / 12.0 * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
    #     dCdt = 1 / 12.0 * p_I_to_C * I - 1 / 7.5 * p_C_to_D * min(Beds(t), C) - max(0, C - Beds(t)) - (1 - p_C_to_D) * \
    #            1 / 6.5 * min(Beds(t), C)
    #     dRdt = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * 1 / 6.5 * min(Beds(t), C)
    #     dDdt = 1 / 7.5 * p_C_to_D * min(Beds(t), C) + max(0, C - Beds(t))
    #
    #     return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt
    #
    #
    # def logistic_R_0(t, R_0_start, k, x0, R_0_end):
    #     return (R_0_start - R_0_end) / (1 + np.exp(-k * (-t + x0))) + R_0_end
    #
    #
    # # SIER-CD Model
    # def Model(days, agegroups, beds_per_100k, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s):
    #
    #     def beta(t):
    #         return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma
    #
    #     N = sum(agegroups)
    #
    #     def Beds(t):
    #         beds_0 = beds_per_100k / 100_000 * N
    #         return beds_0 + s * beds_0 * t  # 0.003
    #
    #     y0 = N - 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    #     t = np.linspace(0, days, days)
    #     ret = odeint(deriv, y0, t, args=(beta, gamma, sigma, N, prob_I_to_C, prob_C_to_D, Beds))
    #     S, E, I, C, R, D = ret.T
    #     R_0_over_time = [beta(i) / gamma for i in range(len(t))]
    #
    #     return t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D
    #
    #
    # def fitter(x, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s):
    #     ret = Model(days, agegroups_us, beds_us, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s)
    #
    #     return ret[6][x]
    #
    #
    # # fitting model
    # mod = lmfit.Model(fitter)
    #
    # for kwarg, (init, mini, maxi) in params_init_min_max.items():
    #     mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)
    #
    # params = mod.make_params()
    # fit_method = "leastsq"
    #
    # result = mod.fit(y_data, params, method="least_squares", x=x_data)
    #
    # np.set_printoptions(suppress=True)
    # # order of output: t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D
    # print(Model(int(y_data.shape[0]+3), agegroups_us, beds_us, **result.best_values)[6])
    # print(y_data)
    #
    # result.plot_fit(datafmt="-", ax=1)
    # plt.show()

    # y_pred = [35122.92628283, 38087.23708003, 41184.46016867]
    # y_test = [36773.0, 38664.0, 40661.0]
    #
    # def scorer(y_pred, y_test):
    #     return np.sum([(np.log10(abs(x)) - np.log10(abs(y)))**2 for x, y in zip(y_pred, y_test)])








    # # Italy - For Cases!
    # cases_italy = cases[cases.region == ('Italy', 'None')]
    # deaths_italy = deaths[deaths.region == ('Italy', 'None')]
    # recovered_italy = recovered[recovered.region == ('Italy', 'None')]
    #
    # covid_italy = []
    # for i in days:
    #     covid_italy.append([int(cases_italy[i]), int(deaths_italy[i]), int(recovered_italy[i])])
    #
    # # data, parameters, etc.
    # covid_italy_cases = np.asarray(list(zip(*covid_italy))[0])
    # beds = pd.read_csv('beds.csv', header=0)
    # beds_lookup = dict(zip(beds["Country"], beds["ICU_Beds"]))
    # beds_italy = beds_lookup['Italy']
    #
    # agegroups = pd.read_csv('agegroups.csv')
    # agegroup_lookup = dict(zip(agegroups['Location'],
    #                            agegroups[['0_9', '10_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80_89',
    #                                       '90_100']].values))
    # agegroups_italy = agegroup_lookup['Italy']
    #
    # probabilities = pd.read_csv('probabilities.csv')
    # prob_I_to_C_1 = list(probabilities.prob_I_to_ICU_1.values)
    # prob_I_to_C_2 = list(probabilities.prob_I_to_ICU_2.values)
    # prob_C_to_Death_1 = list(probabilities.prob_ICU_to_Death_1.values)
    # prob_C_to_Death_2 = list(probabilities.prob_ICU_to_Death_2.values)
    #
    # gamma = 1.0 / 9.0
    # sigma = 1.0 / 3.0
    #
    # # outbreak_shift = np.where(covid_italy_deaths > 0)[0][0]
    # outbreak_shift = 30  # maybe let's try 30, 1, 20?
    #
    # if outbreak_shift >= 0:
    #     y_data = np.concatenate((np.zeros(outbreak_shift), covid_italy_cases))
    # else:
    #     y_data = covid_italy_cases[-outbreak_shift:]
    #
    # days = outbreak_shift + len(covid_italy_cases)
    # x_data = np.linspace(0, days - 1, days, dtype=int)
    #
    # # form: {parameter: (initial guess, minimum value, max value)}
    # params_init_min_max = {'R_0_start': (3.0, 2.0, 5.0),
    #                        'k': (2.5, 0.01, 5.0),
    #                        'x0': (90, 0, 120),
    #                        'R_0_end': (0.9, 0.3, 3.5),
    #                        'prob_I_to_C': (0.05, 0.01, 0.1),
    #                        'prob_C_to_D': (0.5, 0.05, 0.8),
    #                        's': (0.003, 0.001, 0.01)}
    #
    #
    # # rate of change
    # def deriv(y, t, beta, gamma, sigma, N, p_I_to_C, p_C_to_D, Beds):
    #     S, E, I, C, R, D = y
    #
    #     dSdt = -beta(t) * I * S / N
    #     dEdt = beta(t) * I * S / N - sigma * E
    #     dIdt = sigma * E - 1 / 12.0 * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
    #     dCdt = 1 / 12.0 * p_I_to_C * I - 1 / 7.5 * p_C_to_D * min(Beds(t), C) - max(0, C - Beds(t)) - (1 - p_C_to_D) * \
    #            1 / 6.5 * min(Beds(t), C)
    #     dRdt = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * 1 / 6.5 * min(Beds(t), C)
    #     dDdt = 1 / 7.5 * p_C_to_D * min(Beds(t), C) + max(0, C - Beds(t))
    #
    #     return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt
    #
    #
    # def logistic_R_0(t, R_0_start, k, x0, R_0_end):
    #     return (R_0_start - R_0_end) / (1 + np.exp(-k * (-t + x0))) + R_0_end
    #
    #
    # # SIER-CD Model
    # def Model(days, agegroups, beds_per_100k, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s):
    #
    #     def beta(t):
    #         return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma
    #
    #     N = sum(agegroups)
    #
    #     def Beds(t):
    #         beds_0 = beds_per_100k / 100_000 * N
    #         return beds_0 + s * beds_0 * t  # 0.003
    #
    #     y0 = N - 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    #     t = np.linspace(0, days, days)
    #     ret = odeint(deriv, y0, t, args=(beta, gamma, sigma, N, prob_I_to_C, prob_C_to_D, Beds))
    #     S, E, I, C, R, D = ret.T
    #     R_0_over_time = [beta(i) / gamma for i in range(len(t))]
    #
    #     return t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D
    #
    #
    # def fitter(x, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s):
    #     ret = Model(days, agegroups_italy, beds_italy, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s)
    #
    #     return ret[6][x]
    #
    #
    # # fitting model
    # mod = lmfit.Model(fitter)
    #
    # for kwarg, (init, mini, maxi) in params_init_min_max.items():
    #     mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)
    #
    # params = mod.make_params()
    # fit_method = "leastsq"
    #
    # result = mod.fit(y_data, params, method="least_squares", x=x_data)
    #
    # np.set_printoptions(suppress=True)
    # # order of output: t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D
    # print(Model(int(y_data.shape[0] + 3), agegroups_italy, beds_italy, **result.best_values)[6])
    # print(y_data)

    # result.plot_fit(datafmt="-", ax=1, show_init=True)
    # plt.show()











    # plt.plot(covid_us_cases)
    # plt.plot(covid_us_deaths)
    # plt.plot(covid_us_recovered)
    # plt.show()
    # contrived dataset with dependency
    # data = list()
    # for i in range(100):
    #     v1 = i + random()
    #     v2 = v1 + random()
    #     row = [v1, v2]
    #     data.append(row)
    # # fit model
    # print(data)

    #VAR

    # print(covid_us[-5:])
    #
    # covid_us_train = covid_us[:74]
    # covid_us_test = covid_us[-5:]
    #
    # model = VAR(covid_us_train)
    # model_fit = model.fit(3)
    # print(model_fit.summary())
    #
    # # make prediction
    # yhat = model_fit.forecast(model_fit.endog, steps=5)
    # print(yhat)
    #
