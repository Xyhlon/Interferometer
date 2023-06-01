import os

from PyQt5.QtCore import QLibraryInfo
# from PySide2.QtCore import QLibraryInfo

import cv2

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

from scipy.signal import find_peaks
# Poll
I_0_halbe = 765
I_0 = 1169
I_0_halbe = I_0/2 

# s ist in hundertstel mm


from labtool_ex2 import Project
from sympy import exp, pi, sqrt, Abs, pi

# from sympy.physics.units.systems.si import elementary_charge, boltzmann_constant
from scipy.constants import elementary_charge, Boltzmann, h, m_e
import numpy as np

# from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt  # noqa
import os
from uncertainties import ufloat

# pyright: reportUnboundVariable=false
# pyright: reportUndefinedVariable=false


def error(U):
    return U * 0.05 + 0.0010


def luxMeter(I):
    if I<200:
        return 200*0.05
    if I<2000:
        return 2000*0.05


def AmpereLD(x):
    return 0.002


def distanzUnsicherheit(s):
    return 0.0010


def AnodenSpannung(U):
    return 5000 * 0.03


def AmpereTTI(x):
    if x < 0.004:
        return x * 0.001 + 4e-7
    if x < 0.4:
        return x * 0.001 + 4e-5
    if x < 1:
        return x * 0.003 + 1e-3

def doppelSpalt(I_0,lmda,z,d):
    def fun(x):
        return I_0/2*(1+np.cos(2*np.pi*d/(lmda*z)*x))
    return fun

def einzelSpalt(lmda,z,D,I_0=1):
    def fun(x):
        return I_0*np.sinc(x*D/(lmda*z))**2
    return fun


def test_interfero_protokoll():
    gm = {
        "U": r"U",
        "I": r"I",
        "I0": r"I_0",
        "D": r"d",
        "phi": r"\phi",
        "Ev": r"E_v",
        "lmda": r"\lambda",
        "splatbreite": r"d",
        "Deltax": r"\Delta x",
        "Dk1": r"d_\text{K1}",
        "Dk2": r"d_\text{K2}",
        "Dk3": r"d_\text{K3}",
        "Dk4": r"d_\text{K4}",
        "Dk5": r"d_\text{K5}",
        "Dk6": r"d_\text{K6}",
        "Dg1": r"d_\text{G1}",
        "Dg2": r"d_\text{G2}",
        "Dg3": r"d_\text{G3}",
        "Dg4": r"d_\text{G4}",
        "Dg5": r"d_\text{G5}",
        "bigD": r"\bar{d_\text{G}}",
        "smallD": r"\bar{d_\text{K}}",
        "nu": r"\nu",
        "s": r"s",
        "g1": r"g_1",
        "g2": r"g_2",
        "lmbd": r"\lambda",
        "H": r"H",
        "B": r"B",
        "repr": r"\frac{1}{r}",
        "espz": r"e_\text{spez}",
        "c": r"c",
        "k": r"k",
        "N": r"N",
    }
    gv = {
        "lmda": r"\si{\meter}",
        "phi": r"\si{\degree}",
        "Ev": r"\si{\lux}",
        "Deltax": r"\si{\mm}",
        "splatbreite": r"\si{\mm}",
        "I": r"\si{\milli\ampere}",
        "I0": r"\si{\milli\ampere}",
        "s": r"\si{\milli\meter}",
        "nu": r"\si{\mega\hertz}",
        "H": r"\si{\ampere\per\meter}",
        "B": r"\si{\tesla}",
        "Dk1": r"\si{\milli\meter}",
        "Dk2": r"\si{\milli\meter}",
        "Dk3": r"\si{\milli\meter}",
        "Dk4": r"\si{\milli\meter}",
        "Dk5": r"\si{\milli\meter}",
        "Dk6": r"\si{\milli\meter}",
        "Dg1": r"\si{\milli\meter}",
        "Dg2": r"\si{\milli\meter}",
        "Dg3": r"\si{\milli\meter}",
        "Dg4": r"\si{\milli\meter}",
        "Dg5": r"\si{\milli\meter}",
        "bigD": r"\si{\meter}",
        "g1": r"\si{\per\meter}",
        "g2": r"\si{\per\meter}",
        "N": r"1",
        "lmbd": r"\si{\per\meter}",
        "smallD": r"\si{\meter}",
        "repr": r"\si{\per\meter}",
        "espz": r"\si{\coulomb\per\kg}",
        "c": r"\si{\volt\second\per\meter\squared}",
        "k": r"\si{\mega\hertz\per\milli\tesla}",
    }

    pd.set_option("display.max_columns", None)
    plt.rcParams["axes.axisbelow"] = True
    P = Project("Festkoerper", global_variables=gv, global_mapping=gm, font=13)
    P.output_dir = "./"
    P.figure.set_size_inches((10, 4))
    ax: plt.Axes = P.figure.add_subplot()

    D = [
        0.20,
        0.10,
        0.10,
        0.10,
            ]
    d = [
        0.25, 
        0.25, 
        0.50, 
        1.00, 
            ]
    z = 2.51
    lmda = 532e-9
    # I0=1000
    # file = f"./figures/beugungsreferenz.jpg"
    # img_ref = cv2.imread(file)
    # IrefThreeBlocks = np.mean(img_ref,axis=0)
    # # print(IrefThreeBlocks)
    # 
    # # peaks, *_ = find_peaks(, distance=distance, height=height)

    # for i in range(1,5):
    #     file = f"./figures/beugungsbild_spalt{i}_data.jpg"
    #     img = cv2.imread(file,0)
    #     I = np.sum(img, axis=0)
    #     # print(np.argmax(I))
    #     # print(np.mean(I)-I)
    #     distance = 18  # Minimum distance before another peak is searched
    #     height = 230  # height of lowest peak
    #     I_ = np.mean(I)-I
    #     # I = np.mean(img, axis=0)
    #     peaks, *_ = find_peaks(I_, distance=distance, height=height)
    #     peaks = peaks[peaks<200]
    #     # print(peaks)
    #     # print(IrefThreeBlocks.shape)
    #     Iref = np.empty_like(I)
    #     med = np.median(np.diff(peaks,axis=0))*3
    #     # print(np.diff(peaks,axis=0))
    #     # print(med)
    #     # print(np.median(peaks%(med/3)))
    #     BlockRef = cv2.resize(IrefThreeBlocks,(1,int(med)))
    #     # print(BlockRef.shape)
    #     # print(IrefThreeBlocks)
    #     # print(BlockRef.T)
    #     
    #     referenz = np.repeat(BlockRef.T[None,:],int(len(I)//len(BlockRef))+1,axis=0)
    #     referenz =  np.column_stack(referenz)
    #     # print(referenz.shape)
    #     # print(referenz)
    #     Iref = referenz.T[int(np.median(peaks%(med/3))):len(I)+int(np.median(peaks%(med/3)))]
    #     # print(Iref)
    #     # print(Iref.shape)


    #     nump = len(peaks)
    #     pixelPer5mm = (max(peaks)-min(peaks))/nump
    #     pixels = np.arange(len(I)) - np.argmax(I)
    #     millimeters = pixels/pixelPer5mm*5
    #     ax.set_xlabel("$x$ / mm")
    #     # I = np.divide(I,Iref.reshape(len(Iref),))
    #     # print(I.shape)
    #     ax2 = ax
    #     ax = ax.twinx()
    #     print(img.shape)
    #     # ax.plot(millimeters,I, color="#a2f")

    #     # ax.plot(millimeters,Iref)
    #     # ax.scatter(peaks,I_[peaks])

    #     # i=i-1 
    #     I1 = doppelSpalt(I0,lmda,z,d[-i]/1e3)
    #     I2 = einzelSpalt(lmda,z,D[-i]/1e3)
    #     meters = millimeters/1000
    #     # print(meters)
    #     ax.plot(millimeters,I1(meters),color="#f2f",label="Doppelspalt")
    #     ax.plot(millimeters,einzelSpalt(lmda,z,D[-1]/1e3,I0)(meters),color="#a2f",label="Einzelspalt")
    #     I = I1(meters)*I2(meters)
    #     ax.plot(millimeters,I,color="#d20",label="Beide")
    #     # I = I1(meters)*I2(meters-d[-i]/2e3)*I2(meters+d[-i]/2e3)
    #     # ax.plot(millimeters,I,color="#f2f")
    #     # ax.plot(millimeters,Iref,color="#a2f")
    #     ax.set_ylim((0,2000))
    #     ax2.imshow(img, extent=[min(millimeters), max(millimeters), 0,500], alpha=0.9)
    #     P.figure.suptitle(
    #         r"Doppelspalt Inteferenz mit $D=\SI{"+f"{D[-i]}" +r"}{\mm}$ und $d=\SI{"+f"{d[-i]}" +r"}{\mm}$"
    #     )
    #     ax.legend()
    #     P.figure.tight_layout()
    #     # P.ax_legend_all(loc=4)
    #     ax = P.savefig(f"intensity_{i}.pdf")
    #     
    #     # cv2.imshow('image',img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     print(file)

    file = "../data/doppeltspalt.csv"
    filepath = os.path.join(os.path.dirname(__file__), file)
    P.load_data(filepath, loadnew=True)
    print(D[::-1])
    P.data["splatbreite"] = D[::-1]
    P.data["dsplatbreite"] = 0
    P.data["dN"] = 0
    P.data["dDeltax"] = 10
    lmda = Deltax*splatbreite/(1e6*N*z)
    P.resolve(lmda)
    P.print_table(Deltax,N,splatbreite,lmda,name="doppeltspaltMaxMax", inline_units=True,
        options=r"row{1}={font=\mathversion{bold}},",
                  )
    print(P.data)
    mlmda = P.data.u.com["lmda"].mean()
    print(mlmda)
    N = 24
    deltax = ufloat(0.272,0.01)
    print(f"g = {N*z*mlmda/deltax}")

    file = "../data/pol.csv"
    filepath = os.path.join(os.path.dirname(__file__), file)
    P.load_data(filepath, loadnew=True)
    P.data["dphi"] = 2
    P.data["dEv"] = Ev.data.apply(luxMeter)

    P.print_table(phi,Ev,name="pol", inline_units=True,
        options=r"cells={font=\tiny},row{1}={font=\mathversion{bold}\footnotesize},rowsep=0pt,",)


    P.data.phi = phi.data - P.data[P.data["Ev"]==0].phi.values


    P.plot_data(
        ax,
        phi,
        Ev,
        label="Gemessene Daten",
        style="#ad0afd",
        errors=True,
    )

    P.data.phi = np.sort(phi.data)
    
    ax.plot(phi.data,I_0_halbe*np.cos(phi.data*np.pi/180-np.pi/2)**2)

    # filepath = os.path.join(os.path.dirname(__file__), file)
    # P.load_data(filepath, loadnew=True)
    P.figure.suptitle(
        "Gesetzt von Malus\n Winkelabhängigkeit der Intensität zweier Polarisatoren"
    )
    ax = P.savefig("pol.pdf")

    file = "../data/michelson.csv"
    filepath = os.path.join(os.path.dirname(__file__), file)
    P.load_data(filepath, loadnew=True)
    P.vload()
    P.print_table(N,s,name="michelson", inline_units=True,
        options=r"cells={font=\footnotesize},row{1}={font=\mathversion{bold}\footnotesize},",
                  )
    print(2*(s.data.max()-s.data.min())/(N.data.max()-N.data.min())/5.3)
    P.data = P.data.diff()
    x = ufloat(s.data.mean(),s.data.sem())
    print(2*(x)/(5.3*N.data.mean())*1e-5*1e9)

if __name__ == "__main__":
    test_interfero_protokoll()
