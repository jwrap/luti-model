import pickle

import osmnx
from HARMONY_LUTI_SG.globals import *

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import Normalize, LinearSegmentedColormap
import networkx as nx
import osmnx as ox


def plot_map(data, column: str , title: str, save_path: str):
    print("Plotting:", column)
    v_min = data[column].min()
    v_max = data[column].max()

    min_color = 'mediumblue'
    max_color = 'firebrick'
    zero_color = 'white'


    norm = Normalize(vmin=v_min, vmax=v_max)
    cmap_colors = [(norm(v_min), min_color),
                   (norm(0), zero_color),
                   (norm(v_max), max_color)]
    cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_colors)

    fig, ax = plt.subplots(1, figsize=(20, 10))
    data.plot(column=column, cmap=cmap, ax=ax, edgecolor='darkgrey', linewidth=0.1)
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = [data[column].min(), data[column].max()]
    cbar = fig.colorbar(sm)

    # Scalebar
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='upper right', pad=2,
                        border_pad=2)
    ax.add_artist(scalebar)

    # North arrow
    x, y, arrow_length = 0, 1, 0.06
    ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
                ha='center', va='center', fontsize=15, xycoords=ax.transAxes)

    plt.savefig(save_path, dpi=600)

def population_map_creation(inputs, outputs, logger):
    logger.warning("Saving maps...")
    print("Population change.")
    df_popBase = pd.read_csv(outputs['EjOiBase'], usecols= ['zone', 'OiPred_Base'], index_col= 'zone')              # 22
    df_popUSnEBC = pd.read_csv(outputs['EjOiUSnEBC'], usecols=['zone', 'OiPred_USnEBC'], index_col= 'zone')         # S1
    df_popUSEBC = pd.read_csv(outputs['EjOiUSEBC'], usecols=['zone', 'OiPred_USEBC'], index_col= 'zone')            # S2
    df_popUDnEBC = pd.read_csv(outputs['EjOiUDnEBC'], usecols = ['zone', 'OiPred_UDnEBC'], index_col= 'zone')       # S3
    df_popUDEBC = pd.read_csv(outputs['EjOiUDEBC'], usecols=['zone', 'OiPred_UDEBC'], index_col='zone')             # S4
    df_pop_merged = pd.merge(pd.merge(pd.merge(pd.merge(df_popBase, df_popUSnEBC, on = 'zone'),df_popUSEBC, on='zone'),df_popUDnEBC, on='zone'), df_popUDEBC, on='zone')

    df_pop_merged['PopCh_Base_USnEBC'] = ((df_popUSnEBC['OiPred_USnEBC'] - df_popBase['OiPred_Base']) / df_popBase['OiPred_Base']) * 100.0          # base s1
    df_pop_merged['PopCh_Base_USEBC'] = ((df_popUSEBC['OiPred_USEBC'] - df_popBase['OiPred_Base']) / df_popBase['OiPred_Base']) * 100.0             # base s2
    df_pop_merged['PopCH_Base_UDnEBC'] = ((df_popUDnEBC['OiPred_UDnEBC'] - df_popBase['OiPred_Base']) / df_popBase['OiPred_Base']) * 100.0          # base s3
    df_pop_merged['PopCh_Base_UDEBC'] = ((df_popUDEBC['OiPred_UDEBC'] - df_popBase['OiPred_Base']) / df_popBase['OiPred_Base']) * 100.0            # base s4
    df_pop_merged['PopCh_USnEBC_USEBC'] = ((df_popUSEBC['OiPred_USEBC'] - df_popUSnEBC['OiPred_USnEBC']) / df_popUSnEBC['OiPred_USnEBC']) * 100.0   # s1 s2

    df_pop_merged['PopCh_USnEBC_UDnEBC'] = ((df_popUDnEBC['OiPred_UDnEBC'] - df_popUSnEBC['OiPred_USnEBC']) / df_popUSnEBC['OiPred_USnEBC']) * 100.0      # s1 s3
    df_pop_merged['PopCh_USnEBC_UDEBC'] = ((df_popUDEBC['OiPred_UDEBC'] - df_popUSnEBC['OiPred_USnEBC']) / df_popUSnEBC['OiPred_USnEBC']) * 100.0        # s1 s4
    df_pop_merged['PopCh_USEBC_UDnEBC'] = ((df_popUDnEBC['OiPred_UDnEBC'] - df_popUSEBC['OiPred_USEBC']) / df_popUSEBC['OiPred_USEBC']) * 100.0     # s2 s3
    df_pop_merged['PopCh_USEBC_UDEBC'] = ((df_popUDEBC['OiPred_UDEBC'] - df_popUSEBC['OiPred_USEBC']) / df_popUSEBC['OiPred_USEBC']) * 100.0        # s2 s4
    df_pop_merged['PopCh_UDnEBC_UDEBC'] = ((df_popUDEBC['OiPred_UDEBC'] - df_popUDnEBC['OiPred_UDnEBC']) / df_popUDnEBC['OiPred_UDnEBC']) * 100.0
    df_pop_merged.to_csv(Pop_Change)

    pop_ch = pd.read_csv(Pop_Change)
    #
    map_df = gpd.read_file(inputs["DataZonesShapefile"])
    zh_map_popch_df = map_df.merge(pop_ch, left_on='zone', right_on= 'zone')
    # ## Plot population change, prob not needed
    #
    # # # Plotting the Population change between 2019 - 2030
    # fig1, ax1 = plt.subplots(1, figsize=(20, 10))
    # zh_map_popch_df.plot(column='PopCh_Base_USnEBC', cmap='Reds', ax=ax1, edgecolor='darkgrey', linewidth=0.1)
    # ax1.axis('off')
    # ax1.set_title('Population Change Base - Urban Sprawl without E-Bike City in the Zurich Region', fontsize=16)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    # sm._A = [0,zh_map_popch_df['PopCh_Base_USnEBC'].max()]
    # cbar = fig1.colorbar(sm)
    # scalebar = ScaleBar(dx=0.1, label='Scale 1:100000', dimension="si-length", units="m", location='upper right', pad=2,
    #                     border_pad=2)
    # ax1.add_artist(scalebar)
    # x, y, arrow_length = 0, 1, 0.06
    # ax1.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
    #              ha='center', va='center', fontsize=15, xycoords=ax1.transAxes)
    # plt.savefig(outputs["MapPopChangeBaseUSnEBC"], dpi=600)
    # # # plt.show()
    #
    # # # Plotting the Population change between 2030 - 2045
    # fig2, ax2 = plt.subplots(1, figsize=(20, 10))
    # zh_map_popch_df.plot(column='PopCh_Base_USEBC', cmap='Reds', ax=ax2, edgecolor='darkgrey', linewidth=0.1)
    # ax2.axis('off')
    # ax2.set_title('Population Change Base - Urban Sprawl with E-Bike City in the Zurich Region', fontsize=16)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    # sm._A = [0,zh_map_popch_df['PopCh_Base_USEBC'].max()]
    # cbar = fig2.colorbar(sm)
    # scalebar = ScaleBar(dx=0.1, label='Scale 1:100000', dimension="si-length", units="m", location='upper right', pad=2,
    #                     border_pad=2)
    # ax2.add_artist(scalebar)
    # x, y, arrow_length = 0, 1, 0.06
    # ax2.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
    #              ha='center', va='center', fontsize=15, xycoords=ax2.transAxes)
    # plt.savefig(outputs["MapPopChangeBaseUSEBC"], dpi=600)
    # # # plt.show()
    #
    # # # # Plotting the Population change between 2019 - 2045
    # fig3, ax3 = plt.subplots(1, figsize=(20, 10))
    # zh_map_popch_df.plot(column='PopCH_Base_UDnEBC', cmap='Reds', ax=ax3, edgecolor='darkgrey', linewidth=0.1)
    # ax3.axis('off')
    # ax3.set_title('Population Change Base - Urban Densification without E-Bike City in the Zurich Region', fontsize=16)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    # sm._A = [0,zh_map_popch_df['PopCH_Base_UDnEBC'].max()]
    # cbar = fig3.colorbar(sm)
    # scalebar = ScaleBar(dx=0.1, label='Scale 1:100000', dimension="si-length", units="m", location='upper right', pad=2,
    #                     border_pad=2)
    # ax3.add_artist(scalebar)
    # x, y, arrow_length = 0, 1, 0.06
    # ax3.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
    #              ha='center', va='center', fontsize=15, xycoords=ax3.transAxes)
    # plt.savefig(outputs["MapPopChangeBaseUDnEBC"], dpi=600)
    # # # # plt.show()
    # #
    # fig4, ax4 = plt.subplots(1, figsize=(20, 10))
    # zh_map_popch_df.plot(column='PopCh_Base_UDEBC', cmap='Reds', ax=ax4, edgecolor='darkgrey', linewidth=0.1)
    # ax4.axis('off')
    # ax4.set_title('Population Change Base - Urban Densification with E-Bike City in the Zurich Region', fontsize=16)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    # sm._A = [0,zh_map_popch_df['PopCh_Base_UDEBC'].max() ]
    # cbar = fig4.colorbar(sm)
    # scalebar = ScaleBar(dx=0.1, label='Scale 1:100000', dimension="si-length", units="m", location='upper right', pad=2,
    #                     border_pad=2)
    # ax4.add_artist(scalebar)
    # x, y, arrow_length = 0, 1, 0.06
    # ax4.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
    #              ha='center', va='center', fontsize=15, xycoords=ax4.transAxes)
    # plt.savefig(outputs["MapPopChangeBaseUDEBC"], dpi=600)
    # #
    # fig5, ax5 = plt.subplots(1, figsize=(20, 10))
    # zh_map_popch_df.plot(column='PopCh_USnEBC_USEBC', cmap='Reds', ax=ax5, edgecolor='darkgrey', linewidth=0.1)
    # ax5.axis('off')
    # ax5.set_title('Population Change Base - Urban Densification with E-Bike City in the Zurich Region', fontsize=16)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    # sm._A = [zh_map_popch_df['PopCh_USnEBC_USEBC'].min(),zh_map_popch_df['PopCh_USnEBC_USEBC'].max() ]
    # cbar = fig5.colorbar(sm)
    # scalebar = ScaleBar(dx=0.1, label='Scale 1:100000', dimension="si-length", units="m", location='upper right', pad=2,
    #                     border_pad=2)
    # ax5.add_artist(scalebar)
    # x, y, arrow_length = 0, 1, 0.06
    # ax5.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
    #              ha='center', va='center', fontsize=15, xycoords=ax5.transAxes)
    # plt.savefig(outputs["MapPopChangeUSnEBCUSEBC"], dpi=600)
    # #
    # fig6, ax6 = plt.subplots(1, figsize=(20, 10))
    # zh_map_popch_df.plot(column='PopCh_USnEBC_UDnEBC', cmap='Reds', ax=ax6, edgecolor='darkgrey', linewidth=0.1)
    # ax6.axis('off')
    # ax6.set_title('Population Change Urban Sprawl without E-Bike City - Urban Densification without E-Bike City in the Zurich Region', fontsize=16)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    # sm._A = [zh_map_popch_df['PopCh_USnEBC_UDnEBC'].min(),zh_map_popch_df['PopCh_USnEBC_UDnEBC'].max() ]
    # cbar = fig6.colorbar(sm)
    # scalebar = ScaleBar(dx=0.1, label='Scale 1:100000', dimension="si-length", units="m", location='upper right', pad=2,
    #                     border_pad=2)
    # ax6.add_artist(scalebar)
    # x, y, arrow_length = 0, 1, 0.06
    # ax6.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
    #              ha='center', va='center', fontsize=15, xycoords=ax6.transAxes)
    # plt.savefig(outputs["MapPopChangeUSnEBCUDnEBC"], dpi=600)
    # #
    # fig7, ax7 = plt.subplots(1, figsize=(20, 10))
    # zh_map_popch_df.plot(column='PopCh_USnEBC_UDEBC', cmap='Reds', ax=ax7, edgecolor='darkgrey', linewidth=0.1)
    # ax7.axis('off')
    # ax7.set_title('Population Change Urban Sprawl without E-Bike City - Urban Densification with E-Bike City in the Zurich Region', fontsize=16)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    # sm._A = [zh_map_popch_df['PopCh_USnEBC_UDEBC'].min(),zh_map_popch_df['PopCh_USnEBC_UDEBC'].max() ]
    # cbar = fig7.colorbar(sm)
    # scalebar = ScaleBar(dx=0.1, label='Scale 1:100000', dimension="si-length", units="m", location='upper right', pad=2,
    #                     border_pad=2)
    # ax7.add_artist(scalebar)
    # x, y, arrow_length = 0, 1, 0.06
    # ax7.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
    #              ha='center', va='center', fontsize=15, xycoords=ax7.transAxes)
    # plt.savefig(outputs["MapPopChangeUSnEBCUDEBC"], dpi=600)
    # #
    # fig8, ax8 = plt.subplots(1, figsize=(20, 10))
    # zh_map_popch_df.plot(column='PopCh_USEBC_UDnEBC', cmap='Reds', ax=ax8, edgecolor='darkgrey', linewidth=0.1)
    # ax8.axis('off')
    # ax8.set_title(
    #     'Population Change Urban Sprawl with E-Bike City - Urban Densification without E-Bike City in the Zurich Region',
    #     fontsize=16)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    # sm._A = [zh_map_popch_df['PopCh_USEBC_UDnEBC'].min(),zh_map_popch_df['PopCh_USEBC_UDnEBC'].max() ]
    # cbar = fig8.colorbar(sm)
    # scalebar = ScaleBar(dx=0.1, label='Scale 1:100000', dimension="si-length", units="m", location='upper right', pad=2,
    #                     border_pad=2)
    # ax8.add_artist(scalebar)
    # x, y, arrow_length = 0, 1, 0.06
    # ax8.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
    #              ha='center', va='center', fontsize=15, xycoords=ax8.transAxes)
    # plt.savefig(outputs["MapPopChangeUSEBCUDnEBC"], dpi=600)
    # #
    # fig9, ax9 = plt.subplots(1, figsize=(20, 10))
    # zh_map_popch_df.plot(column='PopCh_USEBC_UDEBC', cmap='Reds', ax=ax9, edgecolor='darkgrey', linewidth=0.1)
    # ax9.axis('off')
    # ax9.set_title(
    #     'Population Change Urban Sprawl with E-Bike City - Urban Densification with E-Bike City in the Zurich Region',
    #     fontsize=16)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    # sm._A = [zh_map_popch_df['PopCh_USEBC_UDEBC'].min(),zh_map_popch_df['PopCh_USEBC_UDEBC'].max() ]
    # cbar = fig9.colorbar(sm)
    # scalebar = ScaleBar(dx=0.1, label='Scale 1:100000', dimension="si-length", units="m", location='upper right', pad=2,
    #                     border_pad=2)
    # ax9.add_artist(scalebar)
    # x, y, arrow_length = 0, 1, 0.06
    # ax9.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
    #              ha='center', va='center', fontsize=15, xycoords=ax9.transAxes)
    # plt.savefig(outputs["MapPopChangeUSEBCUDEBC"], dpi=600)
    # #
    # fig10, ax10 = plt.subplots(1, figsize=(20, 10))
    # zh_map_popch_df.plot(column='PopCh_UDnEBC_UDEBC', cmap='Reds', ax=ax10, edgecolor='darkgrey', linewidth=0.1)
    # ax10.axis('off')
    # ax10.set_title(
    #     'Population Change Urban Densification without E-Bike City - Urban Densification with E-Bike City in the Zurich Region',
    #     fontsize=16)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    # sm._A = [zh_map_popch_df['PopCh_UDnEBC_UDEBC'].min(),zh_map_popch_df['PopCh_UDnEBC_UDEBC'].max() ]
    # cbar = fig10.colorbar(sm)
    # scalebar = ScaleBar(dx=0.1, label='Scale 1:100000', dimension="si-length", units="m", location='upper right', pad=2,
    #                     border_pad=2)
    # ax10.add_artist(scalebar)
    # x, y, arrow_length = 0, 1, 0.06
    # ax10.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),
    #              ha='center', va='center', fontsize=15, xycoords=ax10.transAxes)
    # plt.savefig(outputs["MapPopChangeUDnEBCUDEBC"], dpi=600)

    ## Plot Housing Accessibility

    # Housing Accessibility Change
    print("Housing Accessibility Change.")
    df_HA_Base = pd.read_csv(outputs["HousingAccessibilityBase"], usecols=['zone', 'HApuBase', 'HAprBase'])
    df_HA_USnEBC = pd.read_csv(outputs["HousingAccessibilityUSnEBC"], usecols=['zone', 'HApuUSnEBC', 'HAprUSnEBC'])
    df_HA_USEBC = pd.read_csv(outputs["HousingAccessibilityUSEBC"], usecols=['zone', 'HApuUSEBC', 'HAprUSEBC'])
    df_HA_UDnEBC = pd.read_csv(outputs["HousingAccessibilityUDnEBC"], usecols=['zone', 'HApuUDnEBC', 'HAprUDnEBC'])
    df_HA_UDEBC = pd.read_csv(outputs["HousingAccessibilityUDEBC"], usecols=['zone', 'HApuUDEBC', 'HAprUDEBC'])

    # Merging the DataFrames
    df_HA_merged = pd.merge(pd.merge(pd.merge(pd.merge(df_HA_Base, df_HA_USnEBC, on='zone'), df_HA_USEBC, on='zone'),df_HA_UDnEBC, on= 'zone'), df_HA_UDEBC, on= 'zone')

     # public
    df_HA_merged['HACh_Base_USnEBC_pu'] = ((df_HA_USnEBC['HApuUSnEBC'] - df_HA_Base['HApuBase']) / df_HA_Base['HApuBase']) * 100.0
    df_HA_merged['HACh_Base_USEBC_pu'] = ((df_HA_USEBC['HApuUSEBC'] - df_HA_Base['HApuBase']) / df_HA_Base['HApuBase']) * 100.0
    df_HA_merged['HACh_Base_UDnEBC_pu'] = ((df_HA_UDnEBC['HApuUDnEBC'] - df_HA_Base['HApuBase']) / df_HA_Base['HApuBase']) * 100.0
    df_HA_merged['HACh_Base_UDEBC_pu'] = ((df_HA_UDEBC['HApuUDEBC'] - df_HA_Base['HApuBase']) / df_HA_Base['HApuBase']) * 100.0

    df_HA_merged['HACh_USnEBC_USEBC_pu'] = ((df_HA_USEBC['HApuUSEBC'] - df_HA_USnEBC['HApuUSnEBC']) / df_HA_USnEBC['HApuUSnEBC']) * 100.0
    df_HA_merged['HACh_USnEBC_UDnEBC_pu'] = ((df_HA_UDnEBC['HApuUDnEBC'] - df_HA_USnEBC['HApuUSnEBC']) / df_HA_USnEBC['HApuUSnEBC']) * 100.0
    df_HA_merged['HACh_USnEBC_UDEBC_pu'] = ((df_HA_UDEBC['HApuUDEBC'] - df_HA_USnEBC['HApuUSnEBC']) / df_HA_USnEBC['HApuUSnEBC']) * 100.0

    df_HA_merged['HACh_USEBC_UDnEBC_pu'] = ((df_HA_UDnEBC['HApuUDnEBC'] - df_HA_USEBC['HApuUSEBC']) / df_HA_USEBC['HApuUSEBC']) * 100.0
    df_HA_merged['HACh_USEBC_UDEBC_pu'] = ((df_HA_UDEBC['HApuUDEBC']) - df_HA_USEBC['HApuUSEBC'] / df_HA_USEBC['HApuUSEBC']) * 100.0

    df_HA_merged['HACh_UDnEBC_UDEBC_pu'] = ((df_HA_UDEBC['HApuUDEBC'] - df_HA_UDnEBC['HApuUDnEBC']) / df_HA_UDnEBC['HApuUDnEBC']) * 100.0

    # private
    df_HA_merged['HACh_Base_USnEBC_pr'] = ((df_HA_USnEBC['HAprUSnEBC'] - df_HA_Base['HAprBase']) / df_HA_Base['HAprBase']) * 100.0
    df_HA_merged['HACh_Base_USEBC_pr'] = ((df_HA_USEBC['HAprUSEBC'] - df_HA_Base['HAprBase']) / df_HA_Base['HAprBase']) * 100.0
    df_HA_merged['HACh_Base_UDnEBC_pr'] = ((df_HA_UDnEBC['HAprUDnEBC'] - df_HA_Base['HAprBase']) / df_HA_Base['HAprBase']) * 100.0
    df_HA_merged['HACh_Base_UDEBC_pr'] = ((df_HA_UDEBC['HAprUDEBC'] - df_HA_Base['HAprBase']) / df_HA_Base['HAprBase']) * 100.0

    df_HA_merged['HACh_USnEBC_USEBC_pr'] = ((df_HA_USEBC['HAprUSEBC'] - df_HA_USnEBC['HAprUSnEBC']) / df_HA_USnEBC['HAprUSnEBC']) * 100.0
    df_HA_merged['HACh_USnEBC_UDnEBC_pr'] = ((df_HA_UDnEBC['HAprUDnEBC'] - df_HA_USnEBC['HAprUSnEBC']) / df_HA_USnEBC['HAprUSnEBC']) * 100.0
    df_HA_merged['HACh_USnEBC_UDEBC_pr'] = ((df_HA_UDEBC['HAprUDEBC'] - df_HA_USnEBC['HAprUSnEBC']) / df_HA_USnEBC['HAprUSnEBC']) * 100.0

    df_HA_merged['HACh_USEBC_UDnEBC_pr'] = ((df_HA_UDnEBC['HAprUDnEBC'] - df_HA_USEBC['HAprUSEBC']) / df_HA_USEBC['HAprUSEBC']) * 100.0
    df_HA_merged['HACh_USEBC_UDEBC_pr'] = ((df_HA_UDEBC['HAprUDEBC']) - df_HA_USEBC['HAprUSEBC'] / df_HA_USEBC['HAprUSEBC']) * 100.0

    df_HA_merged['HACh_UDnEBC_UDEBC_pr'] = ((df_HA_UDEBC['HAprUDEBC'] - df_HA_UDnEBC['HAprUDnEBC']) / df_HA_UDnEBC['HAprUDnEBC']) * 100.0

    df_HA_merged.to_csv(HA_Change)

    # Plotting the Housing Accessibility change
    HousingAcc_change = pd.read_csv(HA_Change)
    zh_map_HAch_df = map_df.merge(HousingAcc_change, left_on='zone', right_on='zone')
    #

    plot_map(zh_map_HAch_df,'HACh_Base_USnEBC_pu',
             '',
             outputs["MapHousingAccBaseUSnEBCPublic"])
    # # Producing Maps for Housing Accessibility public
    #
    plot_map(zh_map_HAch_df, 'HACh_Base_USEBC_pu',
             '',
             outputs["MapHousingAccBaseUSEBCPublic"])

    plot_map(zh_map_HAch_df, 'HACh_Base_UDnEBC_pu',
             '',
             outputs["MapHousingAccBaseUDnEBCPublic"])
    #
    plot_map(zh_map_HAch_df, 'HACh_Base_UDEBC_pu',
             '',
             outputs["MapHousingAccBaseUDEBCPublic"])
    #
    # plot_map(zh_map_HAch_df, 'HACh_USnEBC_USEBC_pu',
    #          '',
    #          outputs["MapHousingAccUSnEBCUSEBCPublic"])
    # #
    # plot_map(zh_map_HAch_df, 'HACh_USnEBC_UDnEBC_pu',
    #          '',
    #          outputs["MapHousingAccUSnEBCUDnEBCPublic"])
    # #
    # plot_map(zh_map_HAch_df, 'HACh_USnEBC_UDEBC_pu',
    #          '',
    #          outputs["MapHousingAccUSnEBCUDEBCPublic"])
    # # #
    # plot_map(zh_map_HAch_df, 'HACh_USEBC_UDnEBC_pu',
    #          '',
    #          outputs["MapHousingAccUSEBCUDnEBCPublic"])
    # #
    # plot_map(zh_map_HAch_df, 'HACh_USEBC_UDEBC_pu',
    #          '',
    #          outputs["MapHousingAccUSEBCUDEBCPublic"])
    # # #
    # plot_map(zh_map_HAch_df, 'HACh_UDnEBC_UDEBC_pu',
    #          '',
    #          outputs["MapHousingAccUDnEBCUDEBCPublic"])
    # #
    plot_map(zh_map_HAch_df, 'HACh_Base_USnEBC_pr',
             '',
             outputs["MapHousingAccBaseUSnEBCPrivate"])
    # #
    plot_map(zh_map_HAch_df, 'HACh_Base_USEBC_pr',
             '',
             outputs["MapHousingAccBaseUSEBCPrivate"])
    # #
    plot_map(zh_map_HAch_df, 'HACh_Base_UDnEBC_pr',
             '',
             outputs["MapHousingAccBaseUDnEBCPrivate"])
    # #
    plot_map(zh_map_HAch_df, 'HACh_Base_UDEBC_pr',
             '',
             outputs["MapHousingAccBaseUDEBCPrivate"])
    # #
    # plot_map(zh_map_HAch_df, 'HACh_USnEBC_USEBC_pr',
    #          '',
    #          outputs["MapHousingAccUSnEBCUSEBCPrivate"])
    # # #
    # plot_map(zh_map_HAch_df, 'HACh_USnEBC_UDnEBC_pr',
    #          '',
    #          outputs["MapHousingAccUSnEBCUDnEBCPrivate"])
    # # #
    # plot_map(zh_map_HAch_df, 'HACh_USnEBC_UDEBC_pr',
    #          '',
    #          outputs["MapHousingAccUSnEBCUDEBCPrivate"])
    # # #
    # plot_map(zh_map_HAch_df, 'HACh_USEBC_UDnEBC_pr',
    #          '',
    #          outputs["MapHousingAccUSEBCUDnEBCPrivate"])
    # # #
    # plot_map(zh_map_HAch_df, 'HACh_USEBC_UDEBC_pr',
    #          '',
    #          outputs["MapHousingAccUSEBCUDEBCPrivate"])
    # # #
    # plot_map(zh_map_HAch_df, 'HACh_UDnEBC_UDEBC_pr',
    #          '',
    #          outputs["MapHousingAccUDnEBCUDEBCPrivate"])
    # #

    # Jobs Accessibility Change
    print("Jobs Accessibility Change.")
    df_JobAcc_Base = pd.read_csv(outputs["JobsAccessibilityBase"], usecols=['zone', 'JobsApuBase', 'JobsAprBase'])
    df_JobAcc_USnEBC = pd.read_csv(outputs["JobsAccessibilityUSnEBC"], usecols=['zone', 'JobsApuUSnEBC', 'JobsAprUSnEBC'])
    df_JobAcc_USEBC = pd.read_csv(outputs["JobsAccessibilityUSEBC"], usecols=['zone', 'JobsApuUSEBC', 'JobsAprUSEBC'])
    df_JobAcc_UDnEBC = pd.read_csv(outputs['JobsAccessibilityUDnEBC'], usecols=['zone','JobsApuUDnEBC','JobsAprUDnEBC'])
    df_JobAcc_UDEBC = pd.read_csv(outputs['JobsAccessibilityUDEBC'], usecols=['zone','JobsApuUDEBC','JobsAprUDEBC'])

    # Merging the DataFrames
    df_JobAcc_merged = pd.merge(pd.merge(pd.merge(pd.merge(df_JobAcc_Base, df_JobAcc_USnEBC, on='zone'), df_JobAcc_USEBC, on='zone'),df_JobAcc_UDnEBC, on = 'zone'), df_JobAcc_UDEBC, on = 'zone')

    df_JobAcc_merged['JACh_Base_USnEBC_pu'] = ((df_JobAcc_USnEBC['JobsApuUSnEBC'] - df_JobAcc_Base['JobsApuBase']) / df_JobAcc_Base['JobsApuBase']) * 100.0
    df_JobAcc_merged['JACh_Base_USEBC_pu'] = ((df_JobAcc_USEBC['JobsApuUSEBC'] - df_JobAcc_Base['JobsApuBase']) / df_JobAcc_Base['JobsApuBase']) * 100.0
    df_JobAcc_merged['JACh_Base_UDnEBC_pu'] = ((df_JobAcc_UDnEBC['JobsApuUDnEBC'] - df_JobAcc_Base['JobsApuBase']) / df_JobAcc_Base['JobsApuBase']) * 100.0
    df_JobAcc_merged['JACh_Base_UDEBC_pu'] = ((df_JobAcc_UDEBC['JobsApuUDEBC'] - df_JobAcc_Base['JobsApuBase']) / df_JobAcc_Base['JobsApuBase']) * 100.0

    df_JobAcc_merged['JACh_USnEBC_USEBC_pu'] = ((df_JobAcc_USEBC['JobsApuUSEBC'] - df_JobAcc_USnEBC['JobsApuUSnEBC']) / df_JobAcc_USnEBC['JobsApuUSnEBC']) * 100.0
    df_JobAcc_merged['JACh_USnEBC_UDnEBC_pu'] = ((df_JobAcc_UDnEBC['JobsApuUDnEBC'] - df_JobAcc_USnEBC['JobsApuUSnEBC']) / df_JobAcc_USnEBC['JobsApuUSnEBC']) * 100.0
    df_JobAcc_merged['JACh_USnEBC_UDEBC_pu'] = ((df_JobAcc_UDEBC['JobsApuUDEBC'] - df_JobAcc_USnEBC['JobsApuUSnEBC']) / df_JobAcc_USnEBC['JobsApuUSnEBC']) * 100.0

    df_JobAcc_merged['JACh_USEBC_UDnEBC_pu'] = ((df_JobAcc_UDnEBC['JobsApuUDnEBC'] - df_JobAcc_USEBC['JobsApuUSEBC']) / df_JobAcc_USEBC['JobsApuUSEBC']) * 100.0
    df_JobAcc_merged['JACh_USEBC_UDEBC_pu'] = ((df_JobAcc_UDEBC['JobsApuUDEBC']) - df_JobAcc_USEBC['JobsApuUSEBC'] / df_JobAcc_USEBC['JobsApuUSEBC']) * 100.0

    df_JobAcc_merged['JACh_UDnEBC_UDEBC_pu'] = ((df_JobAcc_UDEBC['JobsApuUDEBC'] - df_JobAcc_UDnEBC['JobsApuUDnEBC']) / df_JobAcc_UDnEBC['JobsApuUDnEBC']) * 100.0

    # private
    df_JobAcc_merged['JACh_Base_USnEBC_pr'] = ((df_JobAcc_USnEBC['JobsAprUSnEBC'] - df_JobAcc_Base['JobsAprBase']) / df_JobAcc_Base['JobsAprBase']) * 100.0
    df_JobAcc_merged['JACh_Base_USEBC_pr'] = ((df_JobAcc_USEBC['JobsAprUSEBC'] - df_JobAcc_Base['JobsAprBase']) / df_JobAcc_Base['JobsAprBase']) * 100.0
    df_JobAcc_merged['JACh_Base_UDnEBC_pr'] = ((df_JobAcc_UDnEBC['JobsAprUDnEBC'] - df_JobAcc_Base['JobsAprBase']) / df_JobAcc_Base['JobsAprBase']) * 100.0
    df_JobAcc_merged['JACh_Base_UDEBC_pr'] = ((df_JobAcc_UDEBC['JobsAprUDEBC'] - df_JobAcc_Base['JobsAprBase']) / df_JobAcc_Base['JobsAprBase']) * 100.0

    df_JobAcc_merged['JACh_USnEBC_USEBC_pr'] = ((df_JobAcc_USEBC['JobsAprUSEBC'] - df_JobAcc_USnEBC['JobsAprUSnEBC']) / df_JobAcc_USnEBC['JobsAprUSnEBC']) * 100.0
    df_JobAcc_merged['JACh_USnEBC_UDnEBC_pr'] = ((df_JobAcc_UDnEBC['JobsAprUDnEBC'] - df_JobAcc_USnEBC['JobsAprUSnEBC']) / df_JobAcc_USnEBC['JobsAprUSnEBC']) * 100.0
    df_JobAcc_merged['JACh_USnEBC_UDEBC_pr'] = ((df_JobAcc_UDEBC['JobsAprUDEBC'] - df_JobAcc_USnEBC['JobsAprUSnEBC']) / df_JobAcc_USnEBC['JobsAprUSnEBC']) * 100.0

    df_JobAcc_merged['JACh_USEBC_UDnEBC_pr'] = ((df_JobAcc_UDnEBC['JobsAprUDnEBC'] - df_JobAcc_USEBC['JobsAprUSEBC']) / df_JobAcc_USEBC['JobsAprUSEBC']) * 100.0
    df_JobAcc_merged['JACh_USEBC_UDEBC_pr'] = ((df_JobAcc_UDEBC['JobsAprUDEBC']) - df_JobAcc_USEBC['JobsAprUSEBC'] / df_JobAcc_USEBC['JobsAprUSEBC']) * 100.0

    df_JobAcc_merged['JACh_UDnEBC_UDEBC_pr'] = ((df_JobAcc_UDEBC['JobsAprUDEBC'] - df_JobAcc_UDnEBC['JobsAprUDnEBC']) / df_JobAcc_UDnEBC['JobsAprUDnEBC']) * 100.0

    df_JobAcc_merged.to_csv(Job_Change)

    # Plotting the Jobs Accessibility change
    JobAcc_change = pd.read_csv(Job_Change)
    zh_map_JAch_df = map_df.merge(JobAcc_change, left_on='zone', right_on='zone')


    plot_map(zh_map_JAch_df,
             column='JACh_Base_USnEBC_pu',
             title='',
             save_path=outputs['MapJobsAccBaseUSnEBCPublic'])
    # Producing Maps for Jobs Accessibility Change 2019 - 2030/ 2030 - 2045 / 2019 - 2045 using public/private transport in the Zurich Region
    plot_map(zh_map_JAch_df,
             column='JACh_Base_USEBC_pu',
             title='',
             save_path=outputs['MapJobsAccBaseUSEBCPublic'])
    #
    plot_map(zh_map_JAch_df,
             column='JACh_Base_UDnEBC_pu',
             title='',
             save_path=outputs['MapJobsAccBaseUDnEBCPublic'])
    #
    plot_map(zh_map_JAch_df,
             column='JACh_Base_UDEBC_pu',
             title='',
             save_path=outputs['MapJobsAccBaseUDEBCPublic'])
    #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_USnEBC_USEBC_pu',
    #          title='',
    #          save_path=outputs['MapJobsAccUSnEBCUSEBCPublic'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_USnEBC_UDnEBC_pu',
    #          title='',
    #          save_path=outputs['MapJobsAccUSnEBCUDnEBCPublic'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_USnEBC_UDEBC_pu',
    #          title='',
    #          save_path=outputs['MapJobsAccUSnEBCUDEBCPublic'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_USEBC_UDnEBC_pu',
    #          title='',
    #          save_path=outputs['MapJobsAccUSEBCUDnEBCPublic'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_USEBC_UDEBC_pu',
    #          title='',
    #          save_path=outputs['MapJobsAccUSEBCUDEBCPublic'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_UDnEBC_UDEBC_pu',
    #          title='',
    #          save_path=outputs['MapJobsAccUDnEBCUDEBCPublic'])
    # #
    plot_map(zh_map_JAch_df,
             column='JACh_Base_USnEBC_pr',
             title='',
             save_path=outputs['MapJobsAccBaseUSnEBCPrivate'])
    #
    plot_map(zh_map_JAch_df,
             column='JACh_Base_USEBC_pr',
             title='',
             save_path=outputs['MapJobsAccBaseUSEBCPrivate'])
    #
    plot_map(zh_map_JAch_df,
             column='JACh_Base_UDnEBC_pr',
             title='',
             save_path=outputs['MapJobsAccBaseUDnEBCPrivate'])
    #
    plot_map(zh_map_JAch_df,
             column='JACh_Base_UDEBC_pr',
             title='',
             save_path=outputs['MapJobsAccBaseUDEBCPrivate'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_USnEBC_USEBC_pr',
    #          title='',
    #          save_path=outputs['MapJobsAccUSnEBCUSEBCPrivate'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_USnEBC_UDnEBC_pr',
    #          title='',
    #          save_path=outputs['MapJobsAccUSnEBCUDnEBCPrivate'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_USnEBC_UDEBC_pr',
    #          title='',
    #          save_path=outputs['MapJobsAccUSnEBCUDEBCPrivate'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_USEBC_UDnEBC_pr',
    #          title='',
    #          save_path=outputs['MapJobsAccUSEBCUDnEBCPrivate'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_USEBC_UDEBC_pr',
    #          title='',
    #          save_path=outputs['MapJobsAccUSEBCUDEBCPrivate'])
    # #
    # plot_map(zh_map_JAch_df,
    #          column='JACh_UDnEBC_UDEBC_pr',
    #          title='',
    #          save_path=outputs['MapJobsAccUDnEBCUDEBCPrivate'])
    #



    # Create a common shapefile (polygon) that contains:
    # 1. Population change (2019-2030-2045)
    # 2. Housing Accessibility change pu and pr (2019-2030-2045)
    # 3. Jobs Accessibility change pu and pr (2019-2030-2045)
    tot_shp_df = map_df.merge(pd.merge(pd.merge(HousingAcc_change, JobAcc_change, on='zone'), df_pop_merged, on='zone'), left_on='zone', right_on='zone')
    # Drop unsuseful columns, maybe comment out
    tot_shp_df.drop(columns=['zone','Join_Count','TARGET_FID'], inplace=True, axis=1)
    # Save the shapefile
    tot_shp_df.to_file(outputs["MapResultsShapefile"])

##################################################################################


def flows_map_creation(inputs, outputs, flows_output_keys): # using OSM
    Zone_nodes = nx.read_shp(inputs["ZoneCentroidsShapefileWGS84"])

    Case_Study_Zones = ['Kanton Zürich'] ## set kanton, otherwise only the city is read

    X = nx.graph_from_place(Case_Study_Zones, network_type = 'drive')

    # test plot
    # ox.plot_graph(X)

    X = X.to_undirected()

    # Calculate the origins and destinations for the shortest paths algorithms to be run on OSM graph
    OD_list = calc_shortest_paths_ODs_osm(Zone_nodes, X)

    Flows = []

    for kk, flows_output_keys in enumerate(flows_output_keys):
        Flows.append(pd.read_csv(outputs[flows_output_keys], header=None))

        # Initialise weights to 0
        for source, target in X.edges():
            X[source][target][0]["Flows_" + str(kk)] = 0

    TOT_count = len(OD_list)


    for n, i in enumerate(OD_list):
        print("Flows maps creation - iteration ", n + 1, " of ", TOT_count)
        sssp_paths = nx.single_source_dijkstra_path(X, i,
                                                weight='length')  # single source shortest paths from i to all nodes of the network
        for m, j in enumerate(OD_list):
            shortest_path = sssp_paths[j]  # shortest path from i to j
            path_edges = zip(shortest_path, shortest_path[1:])  # Create edges from nodes of the shortest path

            for edge in list(path_edges):
                for cc in range(len(Flows)):
                    X[edge[0]][edge[1]][0]["Flows_" + str(cc)] += Flows[cc].iloc[n, m]

    # save graph to shapefile
    output_folder_path = "./outputs-Zurich/" + "Flows.shp"
    ox.save_graph_shapefile(X, filepath=output_folder_path)

#############################################################################

def calc_closest(new_node, node_list):
    # Calculate the closest node in the network
    best_diff = 100000000
    closest_node = [0, 0]
    for comp_node in node_list.nodes():

        diff = (abs(comp_node[0] - new_node[0]) + abs(comp_node[1] - new_node[1]))
        if abs(diff) < best_diff:
            best_diff = diff
            closest_node = comp_node

    return closest_node

###############################################################################

def calc_shortest_paths_ODs_osm(zones_centroids, network):
    # For each zone centroid, this function calculates the closest node in the OSM graph.
    # These nodes will be used as origins and destinations in the shortest paths calculations.
    list_of_ODs = []
    for c in zones_centroids:
        graph_clostest_node = ox.nearest_nodes(network, c[0], c[1], return_dist=False)
        list_of_ODs.append(graph_clostest_node)
    return list_of_ODs
