"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This module defines the `Plotting` class, which provides visualization utilities 
    for PyOR simulations.

    The `Plotting` class includes functions for plotting time-domain signals, frequency spectra, 
    evolution of density matrices, and other data relevant to magnetic resonance experiments.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
from matplotlib.widgets import Slider, SpanSelector
from mpl_toolkits.mplot3d import axes3d


class Plotting:
    def __init__(self, class_QS):
        """
        Initialize the Plotting class with default parameters from a configuration object.

        Parameters
        ----------
        class_QS : object
            Configuration class with plotting settings like font size, figure size, limits, etc.
        """
        self.class_QS = class_QS

        self.PlotFigureSize = class_QS.PlotFigureSize
        self.PlotFontSize = class_QS.PlotFontSize
        self.PlotXlimt = class_QS.PlotXlimt
        self.PlotYlimt = class_QS.PlotYlimt
        self.PlotArrowlength = class_QS.PlotArrowlength
        self.PlotLinwidth = class_QS.PlotLinwidth
        self.fig_counter = 1

    def MatrixPlot(self, M, xlabel, ylabel, saveplt = False, savename= "plot"):
        """
        Plot a 2D color map of a matrix with labels.

        Parameters
        ----------
        M : ndarray
            Matrix to be plotted.
        xlabel : list of str
            Labels for x-axis.
        ylabel : list of str
            Labels for y-axis.
        """
        cmap = cm.seismic

        plt.rcParams['figure.figsize'] = self.PlotFigureSize
        plt.rcParams['font.size'] = self.PlotFontSize

        fig = plt.figure(self.fig_counter)
        self.fig_counter += 1
        ax = fig.add_subplot(111)
        cax = ax.matshow(M, interpolation='nearest', cmap=cmap, vmax=abs(M).max(), vmin=-abs(M).max())
        fig.colorbar(cax)

        ax.set_xticks(np.arange(len(xlabel)))
        ax.set_yticks(np.arange(len(ylabel)))
        ax.set_xticklabels(xlabel, rotation='vertical')
        ax.set_yticklabels(ylabel)

        plt.tight_layout()
        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')
        plt.show()

    def MatrixPlot_slider(self, t, rho_t, xlabel, ylabel):
        """
        Plot a time-dependent matrix with a slider to change time steps.

        Parameters
        ----------
        t : ndarray
            Array of time points.
        rho_t : ndarray
            Array of matrices over time (len(t) x N x N).
        xlabel : list of str
            X-axis labels.
        ylabel : list of str
            Y-axis labels.
        """
        cmap = cm.seismic
        plt.rcParams['figure.figsize'] = self.PlotFigureSize
        plt.rcParams['font.size'] = self.PlotFontSize
        plt.rcParams["figure.autolayout"] = True

        fig = plt.figure(self.fig_counter)
        self.fig_counter += 1
        ax = fig.add_subplot(111)
        im = ax.matshow(rho_t[0].real, cmap=cmap)
        cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.65])
        cbar = fig.colorbar(im, cax=cbaxes)

        ax.set_title('T={:.3f}'.format(t[0]))
        ax.set_xticklabels([''] + xlabel)
        ax.set_yticklabels([''] + ylabel)

        fig.subplots_adjust(left=0.25, bottom=0.25)
        axfreq = fig.add_axes([0.2, 0.001, 0.65, 0.03])
        index_slider = Slider(ax=axfreq, label='Time Index', valmin=0, valmax=len(t) - 1, valinit=0, valfmt='%0.0f')

        def update(val):
            index = int(index_slider.val)
            im.set_data(rho_t[index].real)
            ax.set_title('T={:.3f}'.format(t[index]))
            cbar.update_normal(im)
            fig.canvas.draw_idle()

        index_slider.on_changed(update)       
        plt.show()

    def MatrixPlot3D(self, rho, xlabel, ylabel, saveplt=False, savename= "plot"):
        """
        Create a 3D bar plot of matrix values.

        Parameters
        ----------
        rho : ndarray
            Matrix to be plotted (2D).
        xlabel : list of str
            Labels for x-axis.
        ylabel : list of str
            Labels for y-axis.
        """
        plt.rcParams['figure.figsize'] = self.PlotFigureSize
        plt.rcParams['font.size'] = self.PlotFontSize
        rc('font', weight='bold')

        fig = plt.figure(self.fig_counter, constrained_layout=True)
        self.fig_counter += 1
        ax = fig.add_subplot(111, projection='3d')

        num_rows, num_cols = rho.shape
        xpos, ypos = np.meshgrid(np.arange(num_cols) + 0.25, np.arange(num_rows) + 0.25)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)
        dx = dy = 0.5 * np.ones_like(zpos)
        dz = rho.flatten()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', alpha=0.5)
        ax.set_xticks(np.arange(0.5, num_cols, 1))
        ax.set_yticks(np.arange(0.5, num_rows, 1))
        ax.set_xticklabels(xlabel)
        ax.set_yticklabels(ylabel)
        ax.set_zlim(np.min(rho), np.max(rho))
        ax.grid(False)

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')        
        plt.show()

    def Plotting(self, x, y, xlab, ylab, col, saveplt=False, savename= "plot"):
        """
        Plot a simple 2D line graph.

        Parameters
        ----------
        x : array_like
            Array containing data for the x-axis.
        y : array_like
            Array containing data for the y-axis.
        xlab : str
            Label for the x-axis.
        ylab : str
            Label for the y-axis.
        col : str
            Color code or name for the plot line.

        Returns
        -------
        None
        """
        rc('font', weight='bold')
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=self.PlotFigureSize)
        self.fig_counter += 1
        ax1 = fig.add_subplot(111)

        ax1.plot(x, y, linewidth=3.0, color=col)
        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.set_ylabel(ylab, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, linestyle='-.')
        ax1.set_xlim(*self.PlotXlimt)
        ax1.set_ylim(*self.PlotYlimt)

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')        
        plt.show()

    def Plotting_SpanSelector(self, x, y, xlab, ylab, col, saveplt=False, savename= "plot"):
        """
        Plot signal with span selector for interactive region selection.

        This method plots a signal and adds a horizontal span selector tool
        to interactively select a region of the plot. It also displays vertical
        lines marking the selection range and annotates the span width.

        Parameters
        ----------
        x : array_like
            1D array representing the X-axis data.
        y : array_like
            1D array representing the Y-axis data.
        xlab : str
            Label for the X-axis.
        ylab : str
            Label for the Y-axis.
        col : str
            Color code or name for the plot line.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        span_selector : matplotlib.widgets.SpanSelector
            The span selector widget object for interaction.
        """
        rc('font', weight='bold')
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=self.PlotFigureSize)
        self.fig_counter += 1
        spec = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(spec[0, 0])

        ax1.plot(x, y, linewidth=3.0, color=col)
        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.set_ylabel(ylab, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, linestyle='-.')

        xli, xlf = self.PlotXlimt
        yli, ylf = self.PlotYlimt
        ax1.set_xlim(xli, xlf)
        ax1.set_ylim(yli, ylf)

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')

        vline_left = ax1.axvline(0, color='red', linestyle='--', visible=False)
        vline_right = ax1.axvline(0, color='red', linestyle='--', visible=False)
        
        span_text = ax1.text(
            0.05, 0.95, "", transform=ax1.transAxes,
            fontsize=self.PlotFontSize, verticalalignment='top'
        )

        def onselect(xmin, xmax):
            # Update vertical lines
            vline_left.set_xdata([xmin])
            vline_right.set_xdata([xmax])
            vline_left.set_visible(True)
            vline_right.set_visible(True)

            # Update text with span width
            span_text.set_text(f"Selected Span = {xmax - xmin:.4f}")
            fig.canvas.draw_idle()

        span_selector = SpanSelector(ax1, onselect, direction='horizontal', useblit=True)
        
        return fig, span_selector

    def PlottingTwin(self, x, y1, y2, xlab, ylab1, ylab2, col1, col2, saveplt=False, savename= "plot"):
        """
        Plot two signals with twin Y-axes.

        This method generates a plot where `y1` is plotted against `x` on the left Y-axis,
        and `y2` is plotted against `x` on a secondary Y-axis (right), allowing comparison
        of two signals with different scales.

        Parameters
        ----------
        x : array_like
            1D array for the X-axis data.
        y1 : array_like
            1D array for the first Y-axis data (left).
        y2 : array_like
            1D array for the second Y-axis data (right).
        xlab : str
            Label for the X-axis.
        ylab1 : str
            Label for the left Y-axis.
        ylab2 : str
            Label for the right Y-axis.
        col1 : str
            Color for the first plot (y1).
        col2 : str
            Color for the second plot (y2).

        Returns
        -------
        None
            The function displays the plot and does not return anything.
        """
        rc('font', weight='bold')
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=self.PlotFigureSize)
        self.fig_counter += 1
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])
        ax1.plot(x, y1, linewidth=3.0, color=col1)

        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.set_ylabel(ylab1, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.legend(fontsize=self.PlotFontSize, frameon=False)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, linestyle='-.')

        ax2 = ax1.twinx()
        ax2.plot(x, y2, linewidth=3.0, color=col2)

        ax2.set_ylabel(ylab2, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(True, linestyle='-.')

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')

        plt.show()

    def PlottingTwin_SpanSelector(self, x, y1, y2, xlab, ylab1, ylab2, col1, col2, saveplt=False, savename= "plot"):
        """
        Plot two signals with twin Y-axes and a horizontal span selector.

        This function creates a plot with two Y-axes (left and right), allowing visualization
        of two datasets `y1` and `y2` against a common X-axis `x`, each with its own scale.
        A span selector tool is included to highlight and annotate a selected horizontal region.

        Parameters
        ----------
        x : array_like
            1D array representing the X-axis data.
        y1 : array_like
            1D array for the primary Y-axis (left).
        y2 : array_like
            1D array for the secondary Y-axis (right).
        xlab : str
            Label for the X-axis.
        ylab1 : str
            Label for the left Y-axis (corresponding to `y1`).
        ylab2 : str
            Label for the right Y-axis (corresponding to `y2`).
        col1 : str
            Line color for `y1`.
        col2 : str
            Line color for `y2`.

        Returns
        -------
        tuple
            (fig, span_selector), where `fig` is the matplotlib Figure object, and
            `span_selector` is the interactive selector used to highlight a region on the plot.
        """
        rc('font', weight='bold')
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=self.PlotFigureSize)
        self.fig_counter += 1
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])
        ax1.plot(x, y1, linewidth=3.0, color=col1)

        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.set_ylabel(ylab1, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.legend(fontsize=self.PlotFontSize, frameon=False)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, linestyle='-.')

        ax2 = ax1.twinx()
        ax2.plot(x, y2, linewidth=3.0, color=col2)

        ax2.set_ylabel(ylab2, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(True, linestyle='-.')

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')

        # Add interactive span selector and vertical lines
        vline_left = ax2.axvline(0, color='red', linestyle='--', visible=False)
        vline_right = ax2.axvline(0, color='red', linestyle='--', visible=False)

        span_text = ax2.text(0.05, 0.95, "", transform=ax2.transAxes,
                            fontsize=self.PlotFontSize, verticalalignment='top')

        def onselect(xmin, xmax):
            vline_left.set_xdata([xmin])
            vline_right.set_xdata([xmax])
            vline_left.set_visible(True)
            vline_right.set_visible(True)
            span_text.set_text(f"Selected Span = {xmax - xmin:.2f}")
            fig.canvas.draw_idle()

        span_selector = SpanSelector(ax2, onselect, direction='horizontal', useblit=True)

        return fig, span_selector

    def PlottingMulti(self, x, y, xlab, ylab, col, saveplt=False, savename= "plot"):
        """
        Plot multiple signals on a single set of axes.

        Parameters
        ----------
        x : list of array_like
            List of X-axis data arrays, each corresponding to one line.
        y : list of array_like
            List of Y-axis data arrays, each corresponding to one line.
        xlab : str
            Label for the X-axis.
        ylab : str
            Label for the Y-axis.
        col : list of str
            Colors for each plotted line.

        Returns
        -------
        None
            Displays the plot with multiple signals.
        """
        rc('font', weight='bold')
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=self.PlotFigureSize)
        self.fig_counter += 1
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])
        
        for i in range(len(x)):
            ax1.plot(x[i], y[i], linewidth=3.0, color=col[i])

        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.set_ylabel(ylab, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, linestyle='-.')

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')

        plt.show()

    def PlottingMulti_SpanSelector(self, x, y, xlab, ylab, col, saveplt=False, savename= "plot"):
        """
        Plot multiple signals with a horizontal span selector for interactive range selection.

        Parameters
        ----------
        x : list of array_like
            List of X-axis data arrays for each plotted line.
        y : list of array_like
            List of Y-axis data arrays for each plotted line.
        xlab : str
            Label for the X-axis.
        ylab : str
            Label for the Y-axis.
        col : list of str
            List of color values corresponding to each data series.

        Returns
        -------
        tuple
            fig : matplotlib.figure.Figure
                The generated figure object.
            span_selector : matplotlib.widgets.SpanSelector
                The interactive span selector widget.
        """
        rc('font', weight='bold')
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=self.PlotFigureSize)
        self.fig_counter += 1
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])
        
        for i in range(len(x)):
            ax1.plot(x[i], y[i], linewidth=3.0, color=col[i])

        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.set_ylabel(ylab, fontsize=self.PlotFontSize, color='black', fontweight='bold')
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, linestyle='-.')

        vline_left = ax1.axvline(0, color='red', linestyle='--', visible=False)
        vline_right = ax1.axvline(0, color='red', linestyle='--', visible=False)

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')

        span_text = ax1.text(0.05, 0.95, "", transform=ax1.transAxes,
                            fontsize=self.PlotFontSize, verticalalignment='top')

        def onselect(xmin, xmax):
            vline_left.set_xdata([xmin])
            vline_right.set_xdata([xmax])
            vline_left.set_visible(True)
            vline_right.set_visible(True)
            span_text.set_text(f"Selected Span = {xmax - xmin:.2f}")
            fig.canvas.draw_idle()

        span_selector = SpanSelector(ax1, onselect, direction='horizontal', useblit=True)

        return fig, span_selector

    def Plotting3DWire(self, x, y, z, xlab, ylab, title, upL, loL, saveplt=False, savename= "plot"):
        """
        Plot a 3D wireframe surface using meshgrid data.

        Parameters
        ----------
        x : ndarray
            1D array for x-axis values.
        y : ndarray
            1D array for y-axis values.
        z : 2D ndarray
            Matrix of z-values defining the surface height.
        xlab : str
            Label for the x-axis.
        ylab : str
            Label for the y-axis.
        title : str
            Title of the plot.
        upL : float
            Upper limit for the X and Y axes.
        loL : float
            Lower limit for the X and Y axes.

        Returns
        -------
        None
        """
        rc('font', weight='bold')
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=self.PlotFigureSize)
        self.fig_counter += 1
        ax1 = fig.add_subplot(111, projection='3d')

        x1 = x.copy()
        y1 = y.copy()
        x1[(x1 > upL) | (x1 < loL)] = np.nan
        y1[(y1 > upL) | (y1 < loL)] = np.nan

        X, Y = np.meshgrid(x1, y1)
        ax1.plot_wireframe(X, Y, z, lw=0.5, rstride=8, cstride=8)
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)
        ax1.set_title(title)
        ax1.set_xlim3d(loL, upL)
        ax1.set_ylim3d(loL, upL)

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')

        plt.show()

    def PlottingContour(self, x, y, z, xlab, ylab, title, saveplt=False, savename= "plot"):
        """
        Generate a contour plot of a 2D scalar field.

        Parameters
        ----------
        x : ndarray
            1D array of x-axis values.
        y : ndarray
            1D array of y-axis values.
        z : ndarray
            2D array of z values, representing the scalar field.
        xlab : str
            Label for the x-axis.
        ylab : str
            Label for the y-axis.
        title : str
            Title of the contour plot.

        Returns
        -------
        None
        """
        cmap = [cm.RdBu, cm.seismic, cm.bwr, cm.RdGy]
        rc('font', weight='bold')
        fig = plt.figure(self.fig_counter, constrained_layout=True, figsize=self.PlotFigureSize)
        self.fig_counter += 1
        ax1 = fig.add_subplot(111)
        
        plotC = ax1.contour(z, 10, extent=[x.min(), x.max(), y.min(), y.max()],
                            cmap=cmap[1], vmax=abs(z).max(), vmin=-abs(z).max())
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)
        ax1.set_title(title)
        fig.colorbar(plotC)

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')

        plt.show()

    def InnerProduct(self, A, B):
        """
        Calculate the inner product of two matrices or vectors.

        This uses the definition of the inner product as Tr(A†B), where A† is the conjugate transpose of A.

        Parameters
        ----------
        A : ndarray
            First operator or vector.
        B : ndarray
            Second operator or vector.

        Returns
        -------
        complex
            The inner product value as a complex number.
        """
        return np.trace(np.matmul(A.T.conj(), B))
    
    def PlottingSphere(self, Mx, My, Mz, rho_eqQ, plot_vector, scale_datapoints, saveplt=False, savename= "plot"):
        """
        Plot the evolution of magnetization on a Bloch sphere.

        Parameters
        ----------
        Mx : array_like
            Array of Mx components over time.
        My : array_like
            Array of My components over time.
        Mz : array_like
            Array of Mz components over time.
        rho_eqQ : QuantumState
            Equilibrium density matrix wrapped in a custom object.
        plot_vector : bool
            If True, individual magnetization vectors are shown as arrows.
        scale_datapoints : int
            Controls downsampling of the time points shown.

        Returns
        -------
        None
        """
        rho_eq = rho_eqQ.data
        sphera_radius = self.InnerProduct(np.sum(self.class_QS.Sz_, axis=0), rho_eq)

        # Sphere mesh
        phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
        x = sphera_radius * np.sin(theta) * np.cos(phi)
        y = sphera_radius * np.sin(theta) * np.sin(phi)
        z = sphera_radius * np.cos(theta)

        fig = plt.figure(self.fig_counter, figsize=self.PlotFigureSize)
        self.fig_counter += 1
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, color='c', alpha=0.3, rstride=5, cstride=5,
                        linewidth=0.5, edgecolor='k')

        if plot_vector:
            for mx, my, mz in zip(Mx, My, Mz):
                ax.quiver(0, 0, 0, mx, my, mz, color='k', arrow_length_ratio=0.1)

        ax.plot(Mx[::scale_datapoints], My[::scale_datapoints], Mz[::scale_datapoints],
                color='b', linewidth=self.PlotLinwidth)
        ax.quiver(0, 0, 0, Mx[0], My[0], Mz[0], color='r',
                arrow_length_ratio=self.PlotArrowlength, linewidth=self.PlotLinwidth)
        ax.quiver(0, 0, 0, Mx[-1], My[-1], Mz[-1], color='g',
                arrow_length_ratio=self.PlotArrowlength, linewidth=self.PlotLinwidth)

        ax.view_init(10, 20)
        ax.set_xlabel('Mx')
        ax.set_ylabel('My')
        ax.set_zlabel('Mz')

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')

        plt.show()

    def PlottingMultimodeAnalyzer(self, t, freq, sig, spec, saveplt=False, savename= "plot"):
        """
        Multimode Fourier Analyzer with interactive plot linking time and frequency domains.

        Parameters
        ----------
        t : array_like
            Time-domain sampling points.
        freq : array_like
            Frequency-domain sampling points.
        sig : array_like
            Complex-valued signal in the time domain (FID).
        spec : array_like
            Corresponding spectrum of the signal.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The main figure object.
        fourier : Fanalyzer
            An instance of the Fanalyzer class for interaction handling.
        """
        rc('font', weight='bold')
        fig, ax = plt.subplots(2, 2, figsize=self.PlotFigureSize)

        # Top Left: Time domain
        line1, = ax[0, 0].plot(t, sig, "-", color='green')
        ax[0, 0].set_xlabel("Time [s]")
        ax[0, 0].set_ylabel("Signal")
        ax[0, 0].grid()

        # Top Right: Frequency domain
        vline1 = ax[0, 1].axvline(color='k', lw=0.8, ls='--')
        vline2 = ax[0, 1].axvline(color='k', lw=0.8, ls='--')
        text1 = ax[0, 1].text(0.0, 0.0, '', transform=ax[0, 1].transAxes)
        line2, = ax[0, 1].plot(freq, spec, "-", color='green')
        ax[0, 1].set_xlabel("Frequency [Hz]")
        ax[0, 1].set_ylabel("Spectrum")
        ax[0, 1].grid()

        # Bottom Left: Spectrum copy
        line3, = ax[1, 0].plot(freq, spec, "-", color='green')
        ax[1, 0].set_xlabel("Frequency [Hz]")
        ax[1, 0].set_ylabel("Spectrum")
        ax[1, 0].grid()

        # Bottom Right: Reconstructed signal
        vline3 = ax[1, 1].axvline(color='k', lw=0.8, ls='--')
        vline4 = ax[1, 1].axvline(color='k', lw=0.8, ls='--')
        text2 = ax[1, 1].text(0.0, 0.0, '', transform=ax[1, 1].transAxes)
        line4, = ax[1, 1].plot(t, sig, "-", color='green')
        ax[1, 1].set_xlabel("Time [s]")
        ax[1, 1].set_ylabel("Signal")
        ax[1, 1].grid()

        if saveplt:
            plt.savefig(savename + ".pdf", format='pdf')

        # Attach interactivity
        fourier = Fanalyzer(sig.real, sig.imag, ax, fig, line1, line2, line3, line4,
                            vline1, vline2, vline3, vline4, text1, text2)
        fig.canvas.mpl_connect("button_press_event", fourier.button_press)
        fig.canvas.mpl_connect("button_release_event", fourier.button_release)

        return fig, fourier

class Fanalyzer:
    """
    Interactive Fourier analyzer for visualizing time-frequency transformations.

    This class enables interactive selection and analysis of signal portions
    in time and frequency domains. It links two time-domain and two frequency-domain
    plots together, enabling dynamic visual feedback.

    Parameters
    ----------
    Mx : ndarray
        Real part of the signal (time domain).
    My : ndarray
        Imaginary part of the signal (time domain).
    ax : ndarray of Axes
        2x2 array of matplotlib Axes objects.
    fig : matplotlib.figure.Figure
        The figure containing the subplots.
    line1 : Line2D
        Plot handle for time-domain signal (top left).
    line2 : Line2D
        Plot handle for full spectrum (top right).
    line3 : Line2D
        Plot handle for second spectrum (bottom left).
    line4 : Line2D
        Plot handle for reconstructed signal (bottom right).
    vline1, vline2, vline3, vline4 : Line2D
        Vertical lines indicating selection on plots.
    text1, text2 : Text
        Text annotations for selected range (frequency/time).
    """

    def __init__(self, Mx, My, ax, fig, line1, line2, line3, line4,
                 vline1, vline2, vline3, vline4, text1, text2):
        self.x1, self.y1 = line1.get_data()
        self.x2, self.y2 = line2.get_data()
        self.x3, self.y3 = line3.get_data()
        self.x4, self.y4 = line4.get_data()

        self.dt = self.x1[1] - self.x1[0]
        self.fs = 1.0 / self.dt

        self.ax = ax
        self.fig = fig

        self.vline1 = vline1
        self.vline2 = vline2
        self.vline3 = vline3
        self.vline4 = vline4

        self.text1 = text1
        self.text2 = text2

        self.Mx = Mx
        self.My = My
        self.Mt = Mx + 1j * My

    def button_press(self, event):
        """
        Handle mouse press event for interactive selection.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse press event.
        """
        if event.inaxes is self.ax[0, 0]:
            global x1in
            x1in = min(np.searchsorted(self.x1, event.xdata), len(self.x1) - 1)

        elif event.inaxes is self.ax[1, 0]:
            global x3in
            x3in = min(np.searchsorted(self.x3, event.xdata), len(self.x3) - 1)

        elif event.inaxes is self.ax[0, 1]:
            global x2in
            x2in = event.xdata
            self.vline1.set_xdata([x2in])
            plt.draw()

        elif event.inaxes is self.ax[1, 1]:
            global x4in
            x4in = event.xdata
            self.vline3.set_xdata([x4in])
            plt.draw()

    def button_release(self, event):
        """
        Handle mouse release event and update plots based on selection.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse release event.
        """
        if event.inaxes is self.ax[0, 0]:
            global x1fi
            x1fi = min(np.searchsorted(self.x1, event.xdata), len(self.x1) - 1)

            spectrum = np.fft.fft(self.Mt[x1in:x1fi])
            spectrum = np.fft.fftshift(spectrum)
            freq = np.linspace(-self.fs / 2, self.fs / 2, spectrum.shape[-1])

            self.ax[0, 1].get_lines()[-1].remove()
            self.ax[0, 1].plot(self.x2, np.abs(self.y2), "-", color='blue')
            self.ax[0, 1].plot(freq, spectrum, "-", color='red')
            plt.draw()

        elif event.inaxes is self.ax[1, 0]:
            global x3fi
            x3fi = min(np.searchsorted(self.x3, event.xdata), len(self.x3) - 1)

            window = np.zeros_like(self.y3)
            window[x3in:x3fi] = 1.0

            sig = np.fft.ifftshift(self.y3 * window)
            sig = np.fft.ifft(sig)
            t = np.linspace(0, self.dt * self.y3.shape[-1], self.y3.shape[-1])

            self.ax[1, 1].get_lines()[-1].remove()
            self.ax[1, 1].plot(self.x4, self.y4, '-', color='blue')
            self.ax[1, 1].plot(t, sig, '-', color='red')
            plt.draw()

        elif event.inaxes is self.ax[0, 1]:
            global x2fi
            x2fi = event.xdata
            self.vline2.set_xdata([x2fi])
            self.text1.set_text(f'Freq = {abs(x2fi - x2in):.5f} Hz')
            plt.draw()

        elif event.inaxes is self.ax[1, 1]:
            global x4fi
            x4fi = event.xdata
            self.vline4.set_xdata([x4fi])
            self.text2.set_text(f'Time = {abs(x4fi - x4in):.5f} s')
            plt.draw()
