import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import more_itertools as mit

from itertools import groupby
from moviepy.editor import *
from moviepy.video.fx.all import crop
from pathlib import Path
from sklearn.mixture import GaussianMixture
from scipy.ndimage.filters import minimum_filter, maximum_filter, median_filter
from scipy.stats import mode, norm
from scipy.stats import expon

%matplotlib inline


class UC_Freeze:
    def __init__(self, subject_id, pre_processed_csv=None):
        self.subject_id = subject_id
        self.processed_df = None
        self.r_squared_values = None
        self.processed_signal = None
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.start_frame = 0
        self.stop_frame = None
        self.video_duration = None
        self.total_frames = None
        self.processed_frames = 0
        self.GMM = None
        self.freezing_event_durations = None
        self.freezing_bouts = None
        self.total_freezing_time = None

        if pre_processed_csv != None:
            import_pre_processed_data_from_file(pre_processed_csv)


    def get_filenames(self, vid_dir=None, data_dir=None):
        """Summary line.

        Generates video and data filenames and stores them as self.vars for use with other Class methods.

        Parameters
        ----------
        vid_dir : str, defaults to None
            Path to directory of .mov file(s). Filename(s) within this directory should be {subject_id}.mov.
        data_dir : str, defaults to None
            Path to directory of .csv file(s). Filename(s) within this directory should be {subject_id}.csv.

        Returns
        -------
        None

        """
        if vid_dir != None:
            self.my_video_file = os.path.join(vid_dir, f"{self.subject_id}.mov")
            print(self.my_video_file)
        if data_dir != None:
            self.my_data_file = os.path.join(data_dir, f"{self.subject_id}.csv")
            print(self.my_data_file)
        return None


    def manually_set_filenames(self, my_video_file=None, my_data_file=None):
        """Summary line.

        Manually sets video and data filenames and stores them as self.vars for use with other Class methods.

        Parameters
        ----------
        my_video_file : str, defaults to None
            Path to .mov file. Filename should be {subject_id}.mov.
        my_data_file : str, defaults to None
            Path to .csv file. Filename should be {subject_id}.csv.

        Returns
        -------
        None

        """
        if my_video_file != None:
            self.my_video_file = my_video_file
        if my_data_file != None:
            self.my_data_file = my_data_file
        return None


    def import_pre_processed_data_from_file(self, my_data_file, r_squared_col_name=None, processed_col_name=None):
        """Summary line.

        Imports a .csv file containing raw and/or fully processed r_squared values
        reads the file in as a Pandas DataFrame, and stores either/both of the
        values as self.vars for use with other Class methods.

        Parameters
        ----------
        my_data_file : str
            Path to .csv file. Filename should be {subject_id}.csv.
        r_squared_col_name : str, defaults to None
            The column name of the raw r_squared values stored in my_data_file.
        processed_col_name : str, defaults to None
            The column name of the rully processed r_squared values stored in my_data_file.

        Returns
        -------
        None

        """

        self.my_data_file = my_data_file
        my_df = pd.read_csv(self.my_data_file)
        return self.import_pre_processed_data_from_var(my_df, r_squared_col_name=r_squared_col_name, processed_col_name=processed_col_name)


    def import_pre_processed_data_from_var(self, pandas_df, r_squared_col_name=None, processed_col_name=None):
        """Summary line.

        Takes a Pandas DataFrame containing raw and/or fully processed r_squared values
        and stores either/both of the values as self.vars for use with other Class methods.

        Parameters
        ----------
        pandas_df : pandas.core.frame.DataFrame
            A Pandas DataFrame object containing a subject's raw and/or fully processed r_squared values.
        r_squared_col_name : str, defaults to None
            The column name of the raw r_squared values stored in pandas_df.
        processed_col_name : str, defaults to None
            The column name of the rully processed r_squared values stored in pandas_df.

        Returns
        -------
        None

        """
        if all(value is None for value in [r_squared_col_name, processed_col_name]):
            print('You must enter a column name for either/both r_squared_col_name or processed_col_name.')
            return None
        else:
            self.processed_df = pandas_df
            if r_squared_col_name != None:
                self.r_squared_values = self.processed_df[r_squared_col_name].values.reshape(-1,1)
            if processed_col_name != None:
                self.processed_signal = self.processed_df[processed_col_name].values.reshape(-1,1)
            return None


    def set_crop_and_trim_parameters(self, params_file=None, x1=None, x2=None, y1=None, y2=None, start_frame=0, stop_frame=None):
        """Summary line.

        Takes x1 (left edge), x2 (right edge), y1 (bottom edge), and y2 (top edge) arguments to set
        a bounding box for video cropping in other Class methods. These values can be previewed via
        the preview_cropped_frame Class method. Also takes start_frame and stop_frame arguments to
        specify the range of video frames that will be subject to analysis by other Class methods.
        Stores these inputs as self.vars.

        Can accept as the params_file argument a .csv with self.subject_id as row(s) and "x1", "x2",
        "y1", "y2", "start_frame", and "stop_frame" as columns.

        Parameters
        ----------
        params_file : str
            A .csv with "subject_id", "x1", "x2", "y1", "y2", "start_frame", and
            "stop_frame" as columns.
        x1 : int, defaults to None
            The left edge of the bounding box.
        x2 : int, defaults to None
            The right edge of the bounding box.
        y1 : int, defaults to None
            The bottom edge of the bounding box.
        y2 : int, defaults to None
            The top edge of the bounding box.
        start_frame : int, defaults to 0.
            The first of a series of video frames to analyze.
        stop_frame : int, defaults to None
            The last of a series of video frames to analyze.

        Returns
        -------
        None

        """
        if params_file != None:
            df = pd.read_csv(params_file)
            # Create a sub df consisting of the row that corresponds to the current subject:
            sub_df = df.loc[df["subject_id"] == int(self.subject_id)] #########################################################
            # Get that subject's parameters:
            x1 = int(sub_df.iloc[0]["x1"])
            x2 = int(sub_df.iloc[0]["x2"])
            y1 = int(sub_df.iloc[0]["y1"])
            y2 = int(sub_df.iloc[0]["y2"])
            start_frame = int(sub_df.iloc[0]["start_frame"])
            stop_frame = int(sub_df.iloc[0]["stop_frame"])

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.parameters = [x1, x2, y1, y2, start_frame, stop_frame]
        return None
        # best way to handle NaN stop_frame entry?


    def preview_cropped_frame(self, frame=1):
        """Summary line.

        Generates a preview of a single frame cropped to the bounding box dimensions specified using the
        set_crop_and_trim_parameters Class method.

        Parameters
        ----------
        frame : int, defaults to 1.
            Specifies which frame to preview.

        Returns
        -------
        None

        """
        if all(value is None for value in [self.x1, self.x2, self.y1, self.y2]):
            print("Bounding box values have not been set. Set x1, x2, y1, y2 using the set_crop_and_trim_parameters function.")
            return None
        else:
            video_file = VideoFileClip(self.my_video_file)

            # Get the uncropped frame:
            plt.subplot(121)
            preview_frame = []
            preview_frame.append(video_file.get_frame(frame))
            plt.imshow(preview_frame[0])

            # Get the cropped frame:
            plt.subplot(122)
            preview_frame_cropped = []
            video_file_cropped = crop(video_file, self.x1, self.y1, self.x2, self.y2)
            preview_frame_cropped.append(video_file_cropped.get_frame(frame))
            plt.imshow(preview_frame_cropped[0])
            plt.subplots_adjust(bottom=0, right=2, top=1)
            plt.show()
            return None


    def pre_process(self, overwrite=False):#, r2_threshold=0.93, med_filt=9):
        """Summary line.

        Initiates a series of private Class methods that decomposes the self.my_video_file into
        individual frames, converts those frames to grayscale, calculates the r_squared correlation
        between each successive pair of frames, and denoises/smooths the resulting signal.

        Parameters
        ----------
        overwrite : bool, defaults to False.
            If True, allows the user to overwrite self.processed_signal if it is already stored
            as a self.var.
        r2_threshold : float
            Sets the threshold for r_squared outlier correction to be used by the private Class
            method __correct_for_outliers.
        med_filt : int
            Sets the median filter kernel size to be used by the private Class method __denoise_smooth.

        Returns
        -------
        None

        """
        if self.processed_signal is not None:
            if overwrite == False:
                print('Processed Data is already loaded. Set overwrite=True if you wish to overwrite this data.')
                return None

        else:
            if self.r_squared_values is None:
                self.r_squared_values = self.__compute_r_squared_values()
            outlier_corrected = self.__correct_for_outliers()
            denoised = self.__denoise(outlier_corrected)
            median_filtered = self.__median_filter(denoised)
            self.processed_signal = self.__normalize(median_filtered)
            return None


    def __compute_r_squared_values(self, fps=30):
        """Summary line.

        Decomposes the self.my_video_file into individual frames, cropped around the bounding box
        specified via the set_crop_and_trim_parameters Class method; converts those frames to grayscale;
        and calculates the r_squared correlation between each successive pair of frames within the range
        specified via the set_crop_and_trim_parameters Class method.

        Parameters
        ----------
        fps : int, defaults to 30
            Video frames per second.

        Returns
        -------
        r_squared
            A numpy.ndarray of r_squared correlation values.

        """
        if all(value is None for value in [self.x1, self.x2, self.y1, self.y2, self.stop_frame]):
            self.set_crop_and_trim_parameters()
            val = input('Crop and trim parameters can be set via the set_crop_and_trim_parameters Class method. Proceed using defaults? [y/n]')
            if val.lower()[0] == "y":
                pass
            else:
                print("Operation cancelled.")
                return None

        video_file = VideoFileClip(self.my_video_file)
        self.video_duration = int(video_file.duration)
        if all(value is not None for value in [self.x1, self.x2, self.y1, self.y2]):
            video_file_cropped = crop(video_file, x1=self.x1,x2=self.x2,y1=self.y1,y2=self.y2)
            video_to_process = video_file_cropped
        else:
            video_to_process = video_file

        def rgb_to_gray(rgb):
            """Summary line.

            Converts rbg image data to grayscale.

            Parameters
            ----------
            rgb : HxWxN np.array
                An individual frame's rgb data, derived via MoviePy's iter_frames method from its VideoClips class.
                documentation: https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html
            Returns
            -------
            gray
                A grayscale conversion of the HxWxN np.array input.

            """
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray

        # Iterate over the frames, comparing each frame to its preceding frame:
        r_squared = []
        previous_frame = []
        total_frames = 0
        if self.stop_frame == None:
            self.stop_frame = self.video_duration*fps
        else:
            stop_frame = self.stop_frame
        for frame in video_to_process.iter_frames(fps=fps):
            this_frame = rgb_to_gray(frame)
            if previous_frame != []:
                total_frames += 1
                # Limit processing to frames within start_frame/stop_frame range:
                if total_frames <= self.start_frame:
                    pass
                elif total_frames > self.stop_frame:
                    pass
                else:
                    calculate_r_squared = (np.corrcoef(np.ravel(this_frame), np.ravel(previous_frame))[0,1])**2
                    r_squared.append(calculate_r_squared)
            previous_frame = this_frame
        self.processed_frames = len(r_squared)
        self.total_frames = total_frames
        return np.array(r_squared)


    def __correct_for_outliers(self, r_squared_outlier_threshold=0.93):
        """Summary line.

        Iterates over self.r_squared_values and replaces values beneath the specified
        r_squared_outlier_threshold with the mean average of self.r_squared_values.

        Parameters
        ----------
        r_squared_outlier_threshold : float, defaults to 0.93
            The lower boundary of r_squared values to be included in downstream signal
            processing methods.

        Returns
        -------
        outlier_corrected
            A numpy.ndarray of r_squared correlation values corrected following
            outlier correction.

        """
        outlier_corrected = self.r_squared_values
        outlier_corrected[ outlier_corrected < r_squared_outlier_threshold ] = np.mean(self.r_squared_values)
        return outlier_corrected


    def __denoise(self, signal):
        """Summary line.

        Makes denoising corrections to the outlier_corrected signal. First, denoises for
        artificially high spikes in the outlier_corrected signal. Next, median filters the
        resulting signal to denoise for artificially low r_squared values. Next, normalizes
        the resulting signal. Optionally applies a final smoothing operation to ensure that
        slow movements (e.g., of the head) are not spuriously classified as motion.

        Parameters
        ----------
        signal : numpy.ndarray
            An array of r_squared values.

        Returns
        -------
        processed
            A numpy.ndarray of r_squared correlation values corrected following
            outlier correction, smoothing, and normalization.

        """
        # Denoise for artificially high r_squared spikes
        r_squared_threshold = mode(np.round(signal, 5))[0][0]
        r_squared_adjustment = 1.0-r_squared_threshold
        denoised_signal = []
        for i in signal:
            if i > r_squared_threshold:
                denoised_signal.append(1.0)
            else:
                x = i+r_squared_adjustment
                denoised_signal.append(x[0])
        return denoised_signal


    def __median_filter(self, signal, med_filter_kernel=3):

        # Median filter to denoise for artificially low r_squared values:
        med_filtered_signal = []
        med_filtered_signal.append(median_filter(signal, size = med_filter_kernel))
        med_filtered_signal = med_filtered_signal[0]
        return med_filtered_signal


    def __normalize(self, signal):
        norm_signal = [ 1 - ((x - np.min(signal))/(np.max(signal)-np.min(signal))) for x in signal]
        return np.array(norm_signal)


    def gaussian_mix(self, peaks=3, n=300, covar="diag", init="kmeans", t=1e-9, rc=1e-3):
        """Summary line.

        ####### THIS IS A WRAPPER FOR sklearn.mixture.GaussianMixture #######

        Fits self.processed_signal to a 1-dimensional gaussian mixture model. Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

        Parameters
        ----------
        peaks : int
            The number of model components.
        n : int
            The number of EM iterations to perform.
        covar : {'diag' (default), ‘full’, ‘tied’, ‘spherical’}
            Refer to sklearn documentation for details of each option
        init : {‘kmeans’ (default), ‘random’}
            Refer to sklearn documentation for details of each option
        t : float, defaults to 1e-9.
            The convergence threshold. EM iterations will stop when the lower bound
            average gain is below this threshold.
        rc : float, defaults to 1e-3.
            Non-negative regularization added to the diagonal of covariance. Ensures
            that the covariance matrices are all positive.

        Returns
        -------
        None


        """

        data = self.processed_signal.reshape(-1,1)
        self.GMM = GaussianMixture(n_components=peaks, max_iter=n,covariance_type=covar, init_params=init, tol=t, reg_covar=rc).fit(data)
        return None


    def analyze_freezing(self, freezing_threshold=None, med_filter_kernel=3, freezing_duration=3.0):
        """Summary line.

        Derives posterior probability of freezing for every self.processed_signal
        value passed into the Class' gaussian_mix method, uses these probabilities to
        make binary classifications of freezing(1) or motion(0) for each frame, and
        maps groups of consecutive freezing frames to look for freezing events of
        specified duration (traditionally regarded as events of 3 seconds or more).
        Generates key analytical vars self.total_freezing_bouts, self.total_freezing_time,
        and self.activity_factor.

        Parameters
        ----------
        freezing_threshold : float, defaults to None
            The lower bound of posterior probabilities to be treated as freezing during
            framewise classification. Posteriors are derived from the GMM object generated
            by the gaussian_mixture Class method.
        med_filter_kernel : int, defaults to 3.
            Sets the median filter kernel size to smooth the binary freezing and motion
            classification vector.
        freezing_duration : float, defaults to 3.0
            The number of seconds of uninterrupted smoothed binary freezing classificaitons
            required to classify a freezing event.

        Returns
        -------
        None


        """
        freezing_index = np.argmin(self.GMM.means_)
        self.posterior_probabilities = self.GMM.predict_proba(self.processed_signal.reshape(-1, 1))

        # find distributionst that are not freezing, d1, d2
        # compute posterior of 0 for those distributions, post_0_d1, post_0_d2
        #### self.GMM.predict_proba(0)
        # self.p_freezing = posterior_probabilities[:,d1]-post_0_d1 - posterior_probabilities[:,d2]-post_0_d2

        gmm_means = np.sort(subj.GMM.means_)
        zero = np.array([0])
        p_zero = np.sort(subj.GMM.predict_proba(zero.reshape(1, -1) ))

        # THIS IS THE CODE TO COMPUTE FF
        signal_cutoff = self.processed_signal.reshape(-1, 1)[
            np.max( np.where( self.posterior_probabilities[:,2] == np.percentile(self.posterior_probabilities[:,2],
                95, # this is the percentile...
                interpolation='lower') ) )]

        # lamba for 25% mark from link below, is cut-off/ln(4) = exponential_lambda
        # https://en.wikipedia.org/wiki/Exponential_distribution#/media/File:Tukey_anomaly_criteria_for_Exponential_PDF.png
        exponential_lambda = (1 - signal_cutoff)/np.log(4) # This sets the 25% mark of the distribution to signal_cutoff.
        print(exponential_lambda, signal_cutoff, self.processed_signal.reshape(-1, 1).shape )
        self.p_freezing = 1-expon.cdf(1-self.processed_signal.reshape(-1, 1), scale=exponential_lambda, loc=0)

        if freezing_threshold == None:
            freezing_threshold = .9

        #store binary freezing(1) and motion(0) classifications for each frame:
        self.binary_freezing_and_motion = np.zeros_like(self.p_freezing)
        self.binary_freezing_and_motion[ self.p_freezing > freezing_threshold ] = 1

        framewise_freezing_index = [i for (i,j) in enumerate(self.binary_freezing_and_motion) if j == 1]

        #group 90-frame ff events together to identify n-second (or greater) freezing events at 30fps:
        grouping_map = []
        grouped_freezing_frames = []
        compiled_freezing_frames = []

        # Create a grouping map of freezing frames:
        grouping_map = [list(group) for group in mit.consecutive_groups(framewise_freezing_index)]

        # Iterate over the grouping map to scan for events of >= k consecutive freezing frames; i.e., an n-second freezing event:
        self.grouped_freezing_frames = [i for i in grouping_map if len(i) >= freezing_duration*30.0]

        # Flatten grouped_freezing_frames into a list:
        self.compiled_freezing_frames = [j for i in self.grouped_freezing_frames for j in i]

        # Get the durations of freezing events in seconds:
        self.freezing_event_durations = [(max(x)-min(x))/30.0 for x in self.grouped_freezing_frames]

        # Count the total freezing bouts:
        self.freezing_bouts = np.sum(1 for i in self.grouped_freezing_frames if len(i) >= 90)

        # Sum the total ff time in seconds:
        self.total_freezing_time = np.round(np.sum(self.freezing_event_durations), 2)

        # Get activity factor as a general measure of total locomotion:
        self.activity_factor = (self.stop_frame - self.start_frame) - np.sum(self.binary_freezing_and_motion)
        return None


    def plot_denoised_signal(self, xmin=0, xmax=None):
        """Summary line.

        Plots a comparison between the raw and denoised r_squared values,
        self.r_squared_values and self.processed_signal, respectively.

        Parameters
        ----------
        xmin : int, defaults to 0
            The lower bound of the data to plot.
        xmax : int, defaults to None
            The upper bound of the data to plot.

        Returns
        -------
        None


        """
        my_ordered_array = np.arange(0, self.stop_frame - self.start_frame)
        plt.rcParams["figure.figsize"] = (12,4)
#         plt.plot(my_ordered_array, self.r_squared_values, 'k')
        plt.plot(my_ordered_array, self.processed_signal, 'r')
#         plt.gca().invert_yaxis()
        plt.xlim(xmin, xmax)
#         plt.ylim(1.001, min(self.r_squared_values[xmin:xmax]))
        plt.xticks(np.arange(xmin, xmax, step=30), np.arange(0, 100))
        plt.show()
        return None


    def export_csv(self, export_dir, overwrite=False, output="behavior"):
        """Summary line.

        Exports either signal (i.e., framewise) or behavior data for a given subject
        to a .csv file in a specified directory. Behavior data also includes parameters
        set by the user, such as bounding-box coordinates and start/stop frames.

        Parameters
        ----------
        export_dir : str
            A path to the desired export directory.
        overwrite : bool, defaults to False
            If True, allows user to overwrite the existing file associated with
            self.subject_id.
        output : str, ["signal", "behavior"], defaults to "behavior"
            Allows the user to choose between export types. "signal" exports
            framewise signal data including (and derived from) self.r_squared_values,
            whereas "behavior" outputs the results of UC_Freeze's automated
            analyses of those signal data.

        Returns
        -------
        A .csv file of a subject's signal (i.e., framewise) or behavior data.


        """
        f = Path(os.path.join(export_dir, f"{self.subject_id}_{output}.csv"))
        if f.is_file():
            if overwrite == False:
                val = input(f"You are about to overwrite the {output} csv file for subject {self.subject_id}. Proceed? [y/n]")
                if val.lower()[0] == "y":
                    pass
                else:
                    print("Overwrite cancelled.")
                    return None

        if output == "signal":
            dat = [self.r_squared_values, self.processed_signal, self.p_freezing, self.binary_freezing_and_motion]
            cols = ["r2_raw", "r2_processed", "p_ff", "binary_ff"]
        elif output == "behavior":
            dat = [self.x1, self.x2, self.y1, self.y2, self.start_frame, self.stop_frame, sorted(np.concatenate(np.round(self.GMM.means_, 5)).ravel().tolist()),
                   self.freezing_bouts, self.total_freezing_time, np.round((self.total_freezing_time/(0.001+self.freezing_bouts)), 2), self.activity_factor]
            self.behavior_data = dat
            cols = ["x1", "x2", "y1", "y2", "start", "stop", "gmm_peaks", "ff_bouts", "total_ff_time", "avg_ff_time", "activity_factor"]
        else:
            raise ValueError("Invalid output type. Expected one of: signal, behavior")
            return None

        df = pd.DataFrame(data=dat).T
        df.columns = cols
        return df.to_csv(os.path.join(export_dir, f"{self.subject_id}_{output}.csv"))
