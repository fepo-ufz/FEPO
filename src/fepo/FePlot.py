import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors as colors
import logging

class FePlot:
    #__root = "/Volumes/Researches/CHS/CLIMOS/SFFA/resources/data/outcomes/v2/FEPO/"
    __root = "./"
    __save_plot = True
    __name = "FePlot"

    def __init__(self):
        self.ideal_line_color = "#69b3a2"
        self.ground_color = "#69b3a2"
        self.ground_line_color = "#69b3a2"

        self.theme_color_1 = "#B9B7BD"
        self.theme_color_line_1 = "#a8a7a7"

        self.theme_color_2 = "#fec474"
        self.theme_shade_color_2 = "#FEFBE9"
        self.theme_color_line_2 = "#f9ae48"

        self.theme_color_3 = "#abdbe3"
        self.theme_shade_color_3 = "#7EC8E3"
        self.theme_color_line_3 = "#34a0fa"

        self.theme_color_4 = "#90f71b"
        self.theme_shade_color_4 = "#D9FAD9"
        self.theme_color_line_4 = "#469e54"

        self.theme_color_5 = "#0b84a5"
        self.theme_shade_color_5 = "#B3D9E5"
        self.theme_color_line_5 = "#d9d9d9"
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

    def plot_predictions(self, true, pred, outlier=None, title=""):
        if outlier is None:
            outlier = [False] * len(true)

        data = pd.DataFrame({'y_true': true, 'y_pred': pred, 'outlier': outlier})
        data = data.sort_values(by='y_true')

        outlier_colors = {True: self.theme_color_2, False: self.theme_color_3}
        outlier_colors = [outlier_colors[outlier] for outlier in data['outlier']]

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(true)), data['y_pred'], c=outlier_colors, alpha=0.7, s=15)
        plt.plot(range(len(true)), data['y_true'], color=self.theme_color_5, alpha=0.7)
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Log of Density of Sand Flies (Ldt)")
        name = "outlier"
        if self.__save_plot and self.__root is not None:
            plt.savefig(self.__root + name + ".png")
            plt.savefig(self.__root + name + ".pdf")

        plt.show()

    def plot_Ldt_BeforeAfter(self, ldt_before, ldt_after, title=None, name="ldft_before_after"):
        if title is None:
            title = "Density Plot"
        else:
            title = "Density Plot | " + title

        # if length dst_before and dst_after are not equal, then we should remove the extra rows
        min_length = min(len(ldt_before), len(ldt_after))
        ldt_before = ldt_before[:min_length]
        ldt_after = ldt_after[:min_length]

        plt.figure(figsize=(10, 6))
        sns.kdeplot(ldt_before, color=self.theme_color_4, label='Before', linestyle="--")
        sns.kdeplot(ldt_after, color=self.theme_color_5, label='After')
        plt.title(title)
        plt.xlabel('Density')
        plt.ylabel('Frequency')
        plt.legend()

        if (self.__save_plot):
            plt.savefig(self.__root + name + ".png")
            plt.savefig(self.__root + name + ".pdf")

        plt.show()


    def plot_Ldt_BeforeAfter_histogram(self, ldt_before, ldt_after, title=None, nbin= 15, name="ldft_before_after_hist")  :
        if title is None:
            title = "Histogram Plot"
        else:
            title = "Histogram Plot | " + title

        # if length dst_before and dst_after are not equal, then we should remove the extra rows
        min_length = min(len(ldt_before), len(ldt_after))
        ldt_before = ldt_before[:min_length]
        ldt_after = ldt_after[:min_length]

        plt.figure(figsize=(10, 6))
        plt.hist(ldt_before, bins=nbin, alpha=0.5, label='Before')
        plt.hist(ldt_after, bins=nbin, alpha=0.5, label='After')
        plt.legend()
        plt.title(title)
        plt.grid(axis='x', color='lightgray', linestyle='-', linewidth=0.5)

        if (self.__save_plot):
            plt.savefig(self.__root + name + ".png")
            plt.savefig(self.__root + name + ".pdf")

        plt.show()


    def plot_Ldt_distribution_relevancy(self, ldt, relevancy, title=None, name="ldt_relevancy"):

        # Create a figure and left y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the distribution of ldd on the left y-axis
        sns.kdeplot(ldt, color=self.theme_color_4, label='Before', linestyle="--")
        ax1.set_xlabel('Ldt', color=self.theme_color_line_4)
        ax1.set_ylabel('Distribution', color=self.theme_color_line_4)

        # Create a second y-axis on the right
        ax2 = ax1.twinx()

        # Plot relevancy values on the right y-axis
        #ax2.plot(ldt, relevancy, color='red', marker='o')
        ax2.scatter(ldt, relevancy, color=self.theme_color_5, marker='o', s=2)
        ax2.set_ylabel('Relevancy', color=self.theme_color_5)

        # Display the plot
        plt.title('Dual-axis Plot with Matplotlib')
        self.__save_plot = True
        if (self.__save_plot):
            plt.savefig(self.__root + name + ".png")
            plt.savefig(self.__root + name + ".pdf")

        plt.show()

    #define a function plot a barchart such that the x-axis is species name and the y-axis is the number of records
    def plot_species_distribution(self, data, title=None, name="species_distribution"):
        if title is None:
            title = "Species Distribution"
        else:
            title = "Species Distribution | " + title

        species = data['spec'].value_counts()
        species = species.sort_values(ascending=True)
        plt.figure(figsize=(10, 8))
        species.plot(kind='bar', color=self.theme_color_5)
        plt.title(title)
        plt.xlabel('Species')
        plt.ylabel('Number of Records')
        plt.xticks(rotation=65, ha='right', fontsize=8)
        plt.subplots_adjust(bottom=0.35)
        if self.__save_plot:
            plt.savefig(self.__root + name + ".png")
            plt.savefig(self.__root + name + ".pdf")

        plt.show()

    def plot_species_density_distribution(self, data, title=None, name="species_density_distribution"):
        # The x-axis of the plot is the name of species, and the y-axis is the average of dst.
        # If the average of dst is less than 0.09, the bar should be marked with another color.

        if title is None:
            title = "Species Distribution"
        else:
            title = "Species Distribution | " + title

        avg_dst = data.groupby("spec")["dst"].mean()
        avg_dst = avg_dst.sort_values(ascending=True)

        # Define colors based on the condition
        colors = [self.theme_color_2 if val < 0.099 else self.theme_color_5 for val in avg_dst]

        plt.figure(figsize=(10, 8))
        avg_dst.plot(kind='bar', color=colors)
        plt.title(title)
        plt.xlabel('Species')
        plt.ylabel('Average Density of Species')
        plt.xticks(rotation=65, ha='right', fontsize=8)
        plt.subplots_adjust(bottom=0.35)
        if self.__save_plot:
            plt.savefig(self.__root + name + ".png")
            plt.savefig(self.__root + name + ".pdf")

        plt.show()

    def plot_observation_distribution(self, data, name = "observation_distribution"):
        # x-axis is years and y-axis is number of records.
        # sort by year

        sdate_dst = data[['sdate']].copy()
        sdate_dst['sdate'] = pd.to_datetime(sdate_dst['sdate'], errors='coerce')
        sdate_dst = sdate_dst['sdate'].dt.year
        sdate_dst = sdate_dst.value_counts()
        sdate_dst = sdate_dst.sort_index()

        plt.figure(figsize=(10, 6))
        sdate_dst.plot(kind='bar', color=self.theme_color_5)
        plt.title("Observation Distribution")
        plt.xlabel('Year')
        plt.ylabel('Number of Records')

        if self.__save_plot:
            plt.savefig(self.__root + name + ".png")
            plt.savefig(self.__root + name + ".pdf")

    def plot_observation_density_distribution(self, data, name = "observation_density_distribution"):
        # x-axis is years and y-axis is average of dst.
        # sort by year
        # Create a copy of 'sdate' and 'dst' columns
        sdate_dst = data[['sdate', 'dst']].copy()
        sdate_dst['sdate'] = pd.to_datetime(sdate_dst['sdate'], errors='coerce')

        # Group by year and calculate the mean of 'dst'
        avg_dst = sdate_dst.groupby(sdate_dst['sdate'].dt.year)["dst"].mean()
        avg_dst = avg_dst.sort_index()

        plt.figure(figsize=(10, 6))
        avg_dst.plot(kind='bar', color=self.theme_color_5)
        plt.title("Observation Density Distribution")
        plt.xlabel('Year')
        plt.ylabel('Average Density of Sand Flies')

        if self.__save_plot:
            plt.savefig(self.__root + name + ".png")
            plt.savefig(self.__root + name + ".pdf")

    def set_root(self, root):
        self.__root = root
        return self

    def set_name(self, name):
        self.__name = name
        return self

    def set_savePlot(self, savePlot):
        self.__save_plot = savePlot
        return self

    def get_colormap(self, idx, n):

        # Base colors for the gradient
        base_color = self.theme_color_2
        shade_color = self.theme_shade_color_2
        line_color = self.theme_color_line_2
        cmap = colors.LinearSegmentedColormap.from_list("", [shade_color, base_color], N=n)
        map_color = [cmap(i / n) for i in range(n)]
        return map_color
