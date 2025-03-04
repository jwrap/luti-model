import pandas as pd
import statsmodels.api as sm
import numpy as np

def estimate_trips(generation_df, trip_type, predictors, time_step):
    # Estimates trips produced and attracted with a Poisson Regression model.
    # For now, use Poisson only because it yields relatively fewer errors than OLS and Negative Binomial.
    baseline_predictor = [f"BASELINE_{predictor}" for predictor in predictors]

    if time_step == 0: # For the first time step, the predictor is the same as the baseline
        predictors_suffixed = baseline_predictor
        present_predictor = baseline_predictor
    else: # For subsequent time steps, calibrate model with baseline and predict with updated inputs
        present_predictor = [f"{time_step}_{predictor}" for predictor in predictors]
        predictors_suffixed = baseline_predictor + present_predictor

    est_df = generation_df[['ZONE_CODE', f'OBSERVED_{trip_type}'] + predictors_suffixed]

    for _predictor in baseline_predictor: # Always calibrate model with non-zero rows
        est_df = est_df[est_df[_predictor] != 0]

    for _predictor in predictors_suffixed: # Then, take the log-values of the predictors
        est_df[_predictor] = np.log(est_df[_predictor])

    X_spec = est_df[baseline_predictor]
    X_spec = sm.add_constant(X_spec)
    y_spec = est_df[f'OBSERVED_{trip_type}']

    # Specify model using spec data
    poisson_model = sm.GLM(y_spec, X_spec, family=sm.families.Poisson()).fit()

    # Estimate new trips using specified model and updated X data
    X_pred = est_df[present_predictor]
    X_pred = sm.add_constant(X_pred)
    y_pred = poisson_model.predict(X_pred)
    est_df[f'EST_{trip_type}'] = y_pred

    return est_df.drop(columns=predictors_suffixed)

def trip_generation(generation_df, a_predictors, p_predictors, zone_codes, time_step=0):
    attractions_df = estimate_trips(generation_df, 'ATTRACTION', a_predictors, time_step=time_step)
    productions_df = estimate_trips(generation_df, 'PRODUCTION', p_predictors, time_step=time_step)

    df = generation_df[['ZONE_CODE']]
    df = df.merge(attractions_df, on='ZONE_CODE', how='left').merge(productions_df, on='ZONE_CODE', how='left').fillna(
        0)
    df = df.set_index('ZONE_CODE').loc[zone_codes].sort_index()
    scaling_factor = df['EST_ATTRACTION'].sum() / df['EST_PRODUCTION'].sum()
    df['EST_PRODUCTION'] = df['EST_PRODUCTION'] * scaling_factor

    return df['EST_PRODUCTION'], df['EST_ATTRACTION']

def estimate_trips_ss(dat, trip_type, predictors, zone_var, tod='Morning peak'):
	## Estimates trips produced and attracted with a Poisson Regression model.
	# For now, Poisson only because it yielded relatively less errors than OLS and Negative Binomial.

	est_df = dat[[zone_var, f'{trip_type}_{tod}'] + predictors]
	est_df.columns = [zone_var, f'Observed_{trip_type}'] + predictors

	for _predictor in predictors:
		est_df = est_df[est_df[_predictor]!=0]
		est_df[_predictor] = np.log(est_df[_predictor])

	X = est_df[predictors]
	X = sm.add_constant(X)
	y = est_df[f'Observed_{trip_type}']

	poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
	y_pred = poisson_model.predict(X)
	est_df[f'Estimated_{trip_type}'] = y_pred

	return est_df.drop(columns=predictors, axis=1)

def trip_generation_ss(dat, attraction_predictors, production_predictors, zone_codes, tod='Morning peak', zone_var="SUB_MTZ_NO"):
	df = dat[[zone_var]]
	attractions_df = estimate_trips(dat, 'Attraction', attraction_predictors, tod=tod, zone_var=zone_var)
	productions_df = estimate_trips(dat, 'Production', production_predictors, tod=tod, zone_var=zone_var)
	df = df.merge(attractions_df, on=zone_var, how='left').merge(productions_df, on=zone_var, how='left').fillna(0)
	df = df.set_index(zone_var).loc[zone_codes].sort_index()
	scaling_factor = df['Estimated_Attraction'].sum() / df['Estimated_Production'].sum()
	df['Estimated_Production'] = df['Estimated_Production']*scaling_factor

	return df['Estimated_Production'], df['Estimated_Attraction']