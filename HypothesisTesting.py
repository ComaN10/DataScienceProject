import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway, ttest_rel, wilcoxon, kruskal, friedmanchisquare, probplot, shapiro
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

from DataAnalysis import DataAnalysis


class HypothesisTester:
    """
    The t-test assumes that the data is normally distributed and that the variances are equal between groups (for
    unpaired t-test) or within groups (for paired t-test).
    The ANOVA test assumes that the data is normally distributed and that the variances are equal between groups.
    """

    @classmethod
    def test_normal_distribution(cls, groups, group_names) -> bool:
        """
            Tests if the feature is normally distributed validating p_value and statistic
            :param groups:
            :param group_names:
            :return:True if the feature is normally distributed
        """

        cls.qq_plots(group_names, *groups)
        print("Normality tests results (numbers):")
        result_dic = cls.test_normality(group_names, *groups)
        print()

        print("Normality tests results:")
        result = True
        for key, sp_result in result_dic.items():
            # H0 is normal distributed H1 is not normal distributed
            hypothesis_result = not cls.validate_p_value(sp_result[1])
            result = result and hypothesis_result
            print(key, hypothesis_result)

        print()
        result_text = "normal" if result else "not normal"
        print(f"Normality test final result: {result_text}")
        return result

    @classmethod
    def test_hypothesis_single_feature(cls, groups: list, group_names: list, hypothesis: str = "",
                                       homogeneity_variances: bool = True, similar_distribution: bool = True, independence: bool = True,
                                       a: float = 0.05, test_mormality=True, normal: bool = False) -> (bool, bool):
        """
            Uses the corresponding hypothesis tests according to the normality tests.
            :param groups:
            :param group_names:
            :param test_mormality:
            :param normal:if is normal
            :param hypothesis:String with the hypothesis
            :param homogeneity_variances:if groups have a homogeneity variance (== or very close)
            :param similar_distribution:if groups have a similar distribution (== or very close)
            :param independence:The observations between the two groups should also be independent of each other.
            :param a:significance level
            :return:True if hypothesis is H1 (null is to be rejected), False otherwise
        """

        hypothesis = hypothesis if (len(hypothesis) != 0) else "null hypothesis."
        print(f"Testing {hypothesis}")

        normal_test = False
        if (test_mormality and cls.test_normal_distribution(groups, group_names)) or normal:
            normal_test = True
            reject = cls._test_hypothesis_single_feature_normal(groups, group_names, homogeneity_variances, similar_distribution, independence, a)
        else:
            reject = cls._test_hypothesis_single_feature_not_normal(groups, homogeneity_variances, independence, a)

        result = "Reject" if reject else "Not Reject"
        print(f"{result} the {hypothesis}")
        return reject, normal_test

    @classmethod
    def _test_hypothesis_single_feature_normal(cls, groups: list, groups_names: list, homogeneity_variances: bool = True,
                                               similar_distribution: bool = True, independence: bool = True,
                                               a: float = 0.05) -> bool:
        """
            Computes unpaired_anova, paired_anova, unpaired_anova assuming and validates de resulted statistic and p_value.
            By default, that groups have homogeneity_variances and similar_distribution and follows a normal distribution.
            :param groups:List of groups that belong to the same feature but have different classifications
            :param groups_names:
            :param homogeneity_variances:if groups have a homogeneity variance (== or very close)
            :param similar_distribution:if groups have a similar distribution (== or very close)
            :param independence:The observations between the two groups should also be independent of each other.
            :param a:significance level
            :return:True if hypothesis is H1 (null is to be rejected), False otherwise
        """

        if homogeneity_variances:

            try:
                data = pd.DataFrame({
                    'value': np.concatenate(groups),
                    'condition': np.repeat(groups_names, len(groups[0]))
                })
                statistic, p_value = cls.paired_anova(data)
                result = cls.validate_hypothesis("paired_anova", statistic, p_value, a=a)
            except Exception as e:
                print("Error: in paired_anova", e)
                result = True

            if similar_distribution:
                statistic, p_value = cls.unpaired_anova(*groups)
                r = cls.validate_hypothesis("unpaired_anova", statistic, p_value, a=a)
                result = result and r
            else:
                print("Group do not have the necessary pre-requirement similar_distribution for unpaired_anova.")
        else:
            print("Group do not have the necessary pre-requirement homogeneity_variances for paired_anova and unpaired_anova.")
            result = True

        if similar_distribution and independence:
            try:
                statistic, p_value = cls.friedman_test(*groups)
                r = cls.validate_hypothesis("friedman_test", statistic, p_value, a=a)
                result = result and r
            except Exception as e:
                print("Error in friedman_test:", e)
        else:
            print("Group do not have the necessary pre-requirements homogeneity variances and similar_distribution for friedman_test.")

        return result

    @classmethod
    def _test_hypothesis_single_feature_not_normal(cls, groups: list, homogeneity_variances: bool = True,
                                                   independence: bool = True, a: float = 0.05) -> bool:
        """
            Computes kruskal_wallis_test. By default, that groups have homogeneity_variances and independence.
            :param groups:
            :param homogeneity_variances:if groups have a homogeneity variance (== or very close)
            :param independence:if Groups are independent of each other
            :param a:significance level
            :return:True if hypothesis is H1 (null is to be rejected), False otherwise
        """

        if independence and homogeneity_variances:
            statistic, p_value = cls.kruskal_wallis_test(*groups)
            return cls.validate_hypothesis("kruskal_wallis_test", statistic, p_value, a=a)
        else:
            print("Group do not have the necessary pre-requirements homogeneity variances and independence")
            return False

    @classmethod
    def test_hypothesis_between_feature(cls, group1: any, group1_name: str, group2: any, group2_name: str, hypothesis: str = "",
                                        homogeneity_variances: bool = True, independence: bool = True,
                                        a: float = 0.05, test_normality=True, normal:bool = False) -> (bool, bool):
        """
            :param group1:
            :param group1_name:
            :param group2:
            :param group2_name:
            :param test_mormality:
            :param normal:if is normal
            :param hypothesis:String with the null hypothesis
            :param homogeneity_variances:if groups have a homogeneity variance (== or very close)
            :param independence:The observations between the two groups should also be independent of each other.
            :param a:
            :return:True if hypothesis is H1 (null is to be rejected), False otherwise , normality test result
        """

        hypothesis = hypothesis if (len(hypothesis) != 0) else "null hypothesis."
        print(f"Testing {hypothesis}")

        norm_test = False
        if (test_normality and cls.test_normal_distribution([group1, group2], [group1_name, group2_name])) or normal:
            norm_test = True
            reject = cls._test_hypothesis_between_feature_normal(group1, group2, homogeneity_variances, independence, a)
        else:
            reject = True

        r = cls._test_hypothesis_between_feature_independent_of_dist(group1, group2, homogeneity_variances, independence, a)
        reject = reject and r

        result = "Reject" if reject else "Not Reject"
        print(f"{result} the {hypothesis}")

        return reject, norm_test

    @classmethod
    def _test_hypothesis_between_feature_normal(cls, group1: pd.Series, group2: pd.Series,
                                                homogeneity_variances: bool = True, independence: bool = True,
                                                a: float = 0.05) -> bool:
        """
            Computes paired_t_test and unpaired_t_test. By default, that groups have homogeneity variances and independence.
            :param group1:
            :param group2:
            :param homogeneity_variances:if groups have a homogeneity variance (== or very close)
            :independence:The observations between the two groups should also be independent of each other.
            :param a:significance level
            :param min_stc:minium acceptable statistical value
            :return:True if hypothesis is H1 (null is to be rejected), False otherwise
        """

        if homogeneity_variances:
            statistic, p_value = cls.paired_t_test(group1, group2)
            result = cls.validate_hypothesis("paired_t_test", statistic, p_value, a=a)

            if independence:
                statistic, p_value = cls.unpaired_t_test(group1, group2)
                r = cls.validate_hypothesis("unpaired_t_test", statistic, p_value, a=a)
                result = result and r
            else:
                print("Group do not have the necessary pre-requirement independence")

            return result
        else:
            print("Group do not have the necessary pre-requirement homogeneity variances")
            return False

    @classmethod
    def _test_hypothesis_between_feature_independent_of_dist(cls, group1: any, group2: any,
                                               homogeneity_variances: bool = True, independence: bool = True,
                                               a: float = 0.05) -> bool:
        """
            Computes paired_t_test and unpaired_t_test. By default, that groups have homogeneity variances and independence.
            :param group1:
            :param group2:
            :param homogeneity_variances:if groups have a homogeneity variance (== or very close)
            :independence:The observations between the two groups should also be independent of each other.
            :param a:significance level
            :return:True if hypothesis is H1 (null is to be rejected), False otherwise
        """

        if independence:
            statistic, p_value = cls.wilcoxon_ranksum_test(group1, group2)
            reject = cls.validate_hypothesis("wilcoxon_ranksum_test ", statistic, p_value, a=a)
        else:
            print("Groups do not have the necessary pre-requirements for wilcoxon ranksum test")
            reject = True

        if homogeneity_variances:
            statistic, p_value = cls.wilcoxon_signedrank_test(group1, group2)
            r = cls.validate_hypothesis("wilcoxon_signedrank_test", statistic, p_value, a=a)
            reject = reject and r
        else:
            print("Groups do not have the necessary pre-requirements for wilcoxon signedrank test")

        return reject

    @classmethod
    def validate_hypothesis(cls, test_name: str, statistic: float, p_value: float, a=0.05) -> bool:
        """
            :param test_name:
            :param p_value:
            :param statistic:statistic value
            :param a:significance level
            :return:True if validate_p_value and validate_statistical_value are true
        """
        print(f"Test: {test_name}")
        r = cls.validate_p_value(p_value, a)
        result_in_text = "reject" if r else "not reject"
        print(f"std:{statistic} p-value:{p_value} -> result {result_in_text}")
        return r

    @classmethod
    def validate_p_value(cls,p_value, a=0.05) -> bool:
        """
            Compares p_value with the significance level.
            :param p_value:
            :param a:significance_level
            :return Boolean:True (H1) if the null hypothesis is to be rejected
                            or False (H0) if the null hypothesis is not to be rejected
                            If the p-value is less than or equal to α, you can reject the null hypothesis. Similarly,
                            if the null hypothesis is greater than α, you can fail to reject the null hypothesis.


        """
        return p_value <= a

    @classmethod
    def divide_series_in_groups_by_class(cls, data_analise: DataAnalysis, feature: str):
        """
            :param feature:
            :param data_analise:
            :return list of series: Each list element is a group of elements that belongs to the same class
        """

        groups = list()
        groups_names = list()

        for _class in data_analise.target_classes:
            group_data = data_analise.dataset[data_analise.dataset[data_analise.target_name] == _class]
            groups.append(group_data[feature])
            name = f"f:{feature} class:{_class}"
            groups_names.append(name)

        return groups, groups_names

    @classmethod
    def paired_t_test(cls, group1, group2):
        """
        Perform paired t-test for two groups.

        Parameters:
        - group1: List or array-like object containing data for group 1.
        - group2: List or array-like object containing data for group 2.
                  Should have the same length as group1.

        Returns:
        - t_statistic: The calculated t-statistic. on average one less than others.
        the absolute indicates the amount of difference.
        - p_value: The p-value associated with the t-statistic.
        """
        t_statistic, p_value = ttest_rel(group1, group2)
        return t_statistic, p_value

    @classmethod
    def unpaired_t_test(cls, group1, group2):
        """
        Perform unpaired t-test for two groups.

        Parameters:
        - group1: List or array-like object containing data for group 1.
        - group2: List or array-like object containing data for group 2.

        Returns:
        - t_statistic: The calculated t-statistic.on average one less than others.
        the absolute indicates the amount of difference.
        - p_value: The p-value associated with the t-statistic.
        """
        t_statistic, p_value = ttest_ind(group1, group2)
        return t_statistic, p_value

    @classmethod
    def unpaired_anova(cls, *groups):
        """
        Perform unpaired ANOVA for more than two groups.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object.

        Returns:
        - f_statistic: ratio of variability
        - p_value: The p-value associated with the F-statistic.
        """
        f_statistic, p_value = f_oneway(*groups)
        return f_statistic, p_value

    @classmethod
    def paired_anova(cls, data):
        """
        Perform paired ANOVA (repeated measures ANOVA) for more than two groups.

        Parameters:
        - data: Pandas DataFrame containing the data with columns representing different conditions.

        Returns:
        - f_statistic: ratio of variability
        - p_value: The p-value associated with the F-statistic.
        """
        model = ols('value ~ C(condition)', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table['F'][0], anova_table['PR(>F)'][0]

    @classmethod
    def wilcoxon_ranksum_test(cls, group1, group2):
        """
        Perform Wilcoxon rank-sum test (Mann-Whitney U test) for two independent samples.

        Parameters:
        - group1: List or array-like object containing data for sample 1.
        - group2: List or array-like object containing data for sample 2.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = sms.stattools.stats.mannwhitneyu(group1, group2)

        return statistic, p_value

    @classmethod
    def wilcoxon_signedrank_test(cls, group1, group2):
        """
        Perform Wilcoxon signed-rank test for paired samples.
        Defines the alternative hypothesis with ‘greater’ option, this the distribution underlying d is stochastically
        greater than a distribution symmetric about zero; d represent the difference between the paired samples:
        d = x - y if both x and y are provided, or d = x otherwise.

        Parameters:
        - group1: List or array-like object containing data for sample 1.
        - group2: List or array-like object containing data for sample 2.
                  Should have the same length as group1.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = wilcoxon(x=group1, y=group2, alternative="two-sided")
        return statistic, p_value

    @classmethod
    def kruskal_wallis_test(cls, *groups):
        """
        Perform Kruskal-Wallis H test for independent samples.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = kruskal(*groups)
        return statistic, p_value

    @classmethod
    def friedman_test(cls, *groups):
        """
        Perform Friedman test for related samples.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object representing measurements of the same individuals under different conditions.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = friedmanchisquare(*groups)
        return statistic, p_value

    @classmethod
    def qq_plots(cls, variable_names, *data_samples, distribution='norm'):
        """
        Generate Q-Q plots for multiple data samples.

        Parameters:
        - *variable_names: List with the names of the variables to be plotted
        - data_samples: Variable number of 1D array-like objects representing the data samples.
        - distribution: String indicating the theoretical distribution to compare against. Default is 'norm' for normal
        distribution.

        Returns:
        - None (displays the Q-Q plots)
        """
        num_samples = len(data_samples)
        num_rows = (num_samples + 1) // 2  # Calculate the number of rows for subplots
        num_cols = 2 if num_samples > 1 else 1  # Ensure at least 1 column for subplots

        # Generate Q-Q plots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
        axes = axes.flatten()  # Flatten axes if multiple subplots

        for i, data in enumerate(data_samples):
            ax = axes[i]
            probplot(data, dist=distribution, plot=ax)
            ax.set_title(f'Q-Q Plot ({distribution})')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel(variable_names[i])

        # Adjust layout and show plots
        plt.tight_layout()
        plt.show()

    @classmethod
    def test_normality(cls, variable_names, *data_samples):
        """
        Test the normality of multiple data samples using Shapiro-Wilk test.

        Parameters:
        - variable_names: List with the names of the variables to be tested.
        - data_samples: Variable number of 1D array-like objects representing the data samples.

        Returns:
        - results: Dictionary containing the test results for each data sample.
                   The keys are the variable names and the values are a tuple (test_statistic, p_value) for
                   Shapiro-Wilk test.
        """
        results = {}
        for name, data in zip(variable_names, data_samples):
            results[name] = shapiro(data)
        for variable_name, shapiro_result in results.items():
            print(f'{variable_name}:')
            print(f'Shapiro-Wilk test - Test statistic: {shapiro_result.statistic}, p-value: {shapiro_result.pvalue}')
        return results
