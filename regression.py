import pandas as pd  # standard naming convention
import numpy as np  # standard naming convention
import matplotlib.pyplot as plt
import statsmodels.api as sm
import mpld3


DEFAULT_FEATURES = [
    {
        'csv_column': 'BENE_SEX_IDENT_CD',
        'label': 'sex',
        'data_type': 'ordinal',
    },
    {
        'csv_column': 'BENE_AGE_CAT_CD',
        'label': 'age',
        'data_type': 'ordinal',
    },
    {
        'csv_column': 'CC_CANCER',
        'label': 'cancer',
        'data_type': 'ratio',
    },
    {
        'csv_column': 'CC_2_OR_MORE',
        'label': 'multiple_illnesses',
        'data_type': 'ratio',
    },
]

DEFAULT_FEATURE_EXCLUSIONS = ['sex_2']

DEFAULT_TARGET = 'AVE_PA_PAY_PA_EQ_12'

CREATE_DUMMY_TYPES = ['ordinal', 'categorical']


class CSVRegression():
    def __init__(self, csv_file):
        # load data into a pandas dataframe, commonly named df
        self.df = pd.read_csv(csv_file)
        self.source_file = csv_file

    def generate_model_data(self,
                            features=DEFAULT_FEATURES,
                            feature_exclusions=DEFAULT_FEATURE_EXCLUSIONS,
                            target=DEFAULT_TARGET):
        # make a copy of data to build model from
        model_data = self.df.copy()
        # define constants
        model_data['constant'] = np.ones(len(model_data))
        # build features list
        model_features = []

        for feature in features:
            if feature['data_type'] in CREATE_DUMMY_TYPES:
                feature_dummies = pd.get_dummies(model_data[feature['csv_column']],
                                                 prefix=feature['label'])
                model_data = pd.concat([model_data, feature_dummies], axis=1)
                for column in feature_dummies.columns:
                    # don't add any excluded features
                    if column not in feature_exclusions:
                        model_features.append(column)
            else:
                # rename features with specified labels
                if feature['label'] != feature['csv_column']:
                    current_columns = model_data.columns.values.tolist()
                    new_columns = current_columns.copy()
                    column_index = new_columns.index(feature['csv_column'])
                    new_columns[column_index] = feature['label']
                    model_data.rename(columns=dict(zip(current_columns, new_columns)), inplace=True)
                model_features.append(feature['label'])

        # drop rows with null data
        model_data = model_data[model_features + [target]].dropna()

        # rename indices
        model_data.reset_index(inplace = True)
        # define model data, features and target on class
        self.model_data = model_data
        self.features = model_features
        self.target = target
        return model_data

    def run_ols_regression(self):
        # create OLS linear regression
        lm_model = sm.OLS(
            self.model_data[self.target],
            self.model_data[self.features]
            ).fit()
        # add prediction values to model data
        self.model_data['prediction'] = lm_model.fittedvalues
        # add results as class attribute
        self.results = lm_model
        return self.results

    def construct_plot(self, save_template=False, template_name="model.html"):
        # build plot data
        fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
        scatter = ax.scatter(self.model_data[self.target].values,
                             self.model_data['prediction'].values.astype(int),
                             c= 'r',
                             cmap=plt.cm.jet)
        ax.grid(color='white', linestyle='solid')
        ax.set_title("Actual (target) vs. prediction", size=20)
        plt.ylabel('prediction')
        plt.xlabel('actual')

        # create interactive tooltip, save as HTML
        tooltip = mpld3.plugins.PointLabelTooltip(scatter)
        mpld3.plugins.connect(fig, tooltip)
        if save_template:
            mpld3.save_html(fig, template_name)
        return fig

    def generate_summary_dict(self, with_html=True):
        """
        This method will allow us to seamlessly conform
        our regression results into a Django model,
        making an easy workflow to save results
        """
        try:
            # from https://github.com/statsmodels/statsmodels/blob/master/statsmodels/regression/linear_model.py#L1999
            jb, jbpv, skew, kurtosis = sm.stats.stattools.jarque_bera(self.results.wresid)
            self.summary_dict = {
                'source_file': self.source_file,
                'features': ', '.join(self.features),
                'target': self.target,
                'analysis_type': 'OLS Linear Regression',
                'aic': self.results.aic,
                'bic': self.results.bic,
                'num_observations': self.results.nobs,
                'df_residuals': self.results.df_resid,
                'r_squared': self.results.rsquared,
                'r_squared_adjusted': self.results.rsquared_adj,
                'f_statistic': self.results.fvalue,
                'jarque_bera': jb,
                'jarque_bera_prob': jbpv,
                'skew': skew,
                'kurtosis': kurtosis,
                }
            if with_html:
                base_summary_html = self.results.summary().as_html()
                # some styling magic for Bootstrap CSS
                self.summary_dict['summary_html'] = base_summary_html.replace('simpletable', 'table')
                fig = self.construct_plot()
                self.summary_dict['plot_html'] = mpld3.fig_to_html(fig)
            return self.summary_dict
        except AttributeError:
            raise Exception("You must execute `run_ols_regression` method first")


def get_list_value(list_name, index, default):
    try:
        return list_name[index]
    except IndexError:
        return default


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run OLS regressions with CSV data.')
    parser.add_argument('csv_file', type=argparse.FileType('r'),
                        help='the CSV file to be used for analysis')
    parser.add_argument('--feature-columns', type=str, nargs='*', default=[],
                        help='the features to be used for analysis, pass dict as string')
    parser.add_argument('--feature-labels', type=str, nargs='*', default=[],
                        help='the features to be used for analysis, pass dict as string')
    parser.add_argument('--feature-types', type=str, nargs='*', default=[],
                        help='the features to be used for analysis, pass dict as string')
    parser.add_argument('--feature-exclusions', type=str, nargs='*', default=DEFAULT_FEATURE_EXCLUSIONS,
                        help='the features to exclude from analysis')
    parser.add_argument('--target', type=str, nargs='?', default=DEFAULT_TARGET,
                        help='the target to be used for analysis')

    regression_args = parser.parse_args()
    reg = CSVRegression(regression_args.csv_file)

    if len(regression_args.feature_columns) == 0:
        regression_features = DEFAULT_FEATURES
    else:
        regression_features = []
        for index, feature in enumerate(regression_args.feature_columns):
            regression_features.append({
                'csv_column': feature,
                'label': get_list_value(regression_args.feature_labels, index, feature),
                'data_type': get_list_value(regression_args.feature_types, index, "ratio"),
                })
    reg.generate_model_data(features=regression_features,
                            feature_exclusions=regression_args.feature_exclusions,
                            target=regression_args.target)
    reg.run_ols_regression()
    print(reg.results.summary())