"""
Export every number that Figure 1 (v14) plots, as CSV tables.

Runs the data-preparation half of Figure_1_Mpox_Nigeria_v14_sitrep.py (everything
above the plotting section), then writes one CSV per panel plus a source table.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = '/mnt/user-data/outputs/figure1_data'
os.makedirs(OUTDIR, exist_ok=True)

src = open(os.path.join(HERE, 'Figure_1_Mpox_Nigeria_v14_sitrep.py')).read()
prep = src.split('# ================================================================ figure')[0]
ns = {'__file__': os.path.join(HERE, 'Figure_1_Mpox_Nigeria_v14_sitrep.py')}
exec(compile(prep, 'v14_prep', 'exec'), ns)

WAVES, WAVE_YEARS, WAVE_SPAN = ns['WAVES'], ns['WAVE_YEARS'], ns['WAVE_SPAN']
ZONE_ORDER, STATE_ZONE = ns['ZONE_ORDER'], ns['STATE_ZONE']
wst, lin, old, seq = ns['wst'], ns['lin'], ns['old'], ns['seq']
ANNUAL, AGE_BY_YEAR, STATE_BY_YEAR = ns['ANNUAL'], ns['AGE_BY_YEAR'], ns['STATE_BY_YEAR']
band_pct, age_n = ns['band_pct'], ns['age_n']
full_totals, states_conf, states_susp = ns['full_totals'], ns['states_conf'], ns['states_susp']
zone_wave, zone_pct = ns['zone_wave'], ns['zone_pct']
MONTHLY_2023 = ns['MONTHLY_2023_CONFIRMED']
wlab = {w: w.replace('\n', ' ') for w in WAVES}


def write(df, name, index=False):
    path = os.path.join(OUTDIR, name)
    df.to_csv(path, index=index)
    print(f'{name:46s} {len(df):5d} rows')
    return path


# ---- 00: wave-level summary -------------------------------------------------
summary = pd.DataFrame([{
    'wave': wlab[w],
    'period': f'{WAVE_SPAN[w][0]} to {WAVE_SPAN[w][1]}',
    'suspected': full_totals[w][0],
    'confirmed': full_totals[w][1],
    'pct_confirmed_of_suspected': round(100 * full_totals[w][1] / full_totals[w][0], 1),
    'states_with_confirmed_of_37': int(states_conf[w]),
    'states_reporting_suspected_of_37': int(states_susp[w]),
    'northern_zones_pct_of_confirmed': round(zone_pct.loc[w, ['North Central', 'North West', 'North East']].sum(), 1),
    'highest_burden_zone': zone_pct.loc[w].idxmax(),
    'highest_burden_zone_pct': round(zone_pct.loc[w].max(), 1),
    'pct_confirmed_aged_le20': round(ns['le20'][w], 1),
    'pct_confirmed_aged_21_40': round(ns['a2140'][w], 1),
    'source': ('NCDC sitrep wk52 2023' if w != WAVES[2] else 'NRL national line lists 2024-2025'),
} for w in WAVES])
write(summary, '00_wave_summary.csv')

# ---- 01: panel a, monthly cases by zone -------------------------------------
rows = []
for (dt, z), g in old.groupby(['date', 'zone'], dropna=True):
    rows.append((dt.date(), z, int(g['confirmed'].sum()), int(g['nonconfirmed'].sum()),
                 'NCDC aggregate state counts'))
for (dt, z), g in lin.groupby(['month', 'zone'], dropna=True):
    rows.append((dt.date(), z, int(g['confirmed'].sum()), int((~g['confirmed']).sum()),
                 'NRL line list (specimen collection date)'))
monthly = pd.DataFrame(rows, columns=['month', 'zone', 'confirmed', 'suspected_not_confirmed', 'source'])
m23 = pd.DataFrame({
    'month': pd.date_range('2023-01-01', '2023-12-01', freq='MS').date,
    'zone': 'all zones (no state breakdown)',
    'confirmed': MONTHLY_2023,
    'suspected_not_confirmed': np.nan,
    'source': 'NCDC sitrep wk52 2023, Fig. 4',
})
monthly = pd.concat([monthly, m23], ignore_index=True).sort_values(['month', 'zone'])
write(monthly, '01_panelA_monthly_cases_by_zone.csv')

# ---- 02: panel a lower track, genomes per month -----------------------------
gen = (seq.groupby([seq['month'].dt.date, 'zone']).size().rename('genomes')
       .reset_index().rename(columns={'month': 'month'}))
write(gen, '02_panelA_genomes_per_month_by_zone.csv')

# ---- 03: panel b, state-level counts by wave --------------------------------
states = wst.copy()
states['wave'] = states['wave'].map(wlab)
states = states.rename(columns={'state_norm': 'state', 'share_conf': 'pct_of_wave_confirmed',
                                'share_susp': 'pct_of_wave_suspected'})
states['pct_positive'] = (100 * states['confirmed'] / states['suspected'].replace(0, np.nan)).round(1)
states[['pct_of_wave_confirmed', 'pct_of_wave_suspected']] = \
    states[['pct_of_wave_confirmed', 'pct_of_wave_suspected']].round(2)
states = states[['wave', 'zone', 'state', 'suspected', 'confirmed',
                 'pct_of_wave_suspected', 'pct_of_wave_confirmed', 'pct_positive']]
write(states.sort_values(['wave', 'confirmed'], ascending=[True, False]),
      '03_panelB_state_counts_by_wave.csv')

# ---- 04: panel c, zone composition by wave ----------------------------------
zc = zone_wave.copy(); zc.index = [wlab[w] for w in zc.index]
zp = zone_pct.copy(); zp.index = [wlab[w] for w in zp.index]
zone_out = (zc.stack().rename('confirmed').reset_index()
            .rename(columns={'level_0': 'wave', 'zone': 'zone'}))
zone_out['pct_of_wave_confirmed'] = zp.stack().values.round(1)
zone_out['confirmed'] = zone_out['confirmed'].astype(int)
write(zone_out, '04_panelC_zone_composition_by_wave.csv')

# ---- 05: panel d, age composition by wave -----------------------------------
age_counts = pd.DataFrame({wlab[w]: AGE_BY_YEAR[WAVE_YEARS[w]].sum(axis=1) for w in WAVES[:2]})
conf3 = lin[lin['confirmed'] & lin['age_years'].notna()]
w3_counts = pd.cut(conf3['age_years'], bins=[0, 11, 21, 31, 41, 51, np.inf],
                   labels=ns['AGE_BANDS_PUB'], right=False).value_counts()
age_counts[wlab[WAVES[2]]] = w3_counts.reindex(ns['AGE_BANDS_PUB']).fillna(0).astype(int)
age_out = age_counts.reset_index().rename(columns={'index': 'age_band', 'band': 'age_band'})
age_long = age_out.melt(id_vars='age_band', var_name='wave', value_name='confirmed')
age_long['pct_of_wave_confirmed'] = age_long.apply(
    lambda r: round(100 * r['confirmed'] / age_n[{v: k for k, v in wlab.items()}[r['wave']]], 1), axis=1)
write(age_long, '05_panelD_age_composition_by_wave.csv')

# ---- 06: panel e, wave 3 age-sex pyramid ------------------------------------
age_bins = [0, 5, 10, 15, 20, 30, 40, 50, 60, np.inf]
age_labels = ['0-4', '5-9', '10-14', '15-19', '20-29', '30-39', '40-49', '50-59', '60+']
lin['age_group'] = pd.cut(lin['age_years'], bins=age_bins, labels=age_labels, right=False)
pyr = lin[lin['sex'].isin(['MALE', 'FEMALE']) & lin['age_group'].notna()]
pyramid = (pyr.groupby(['age_group', 'sex', 'confirmed'], observed=True).size()
           .rename('n').reset_index())
pyramid['status'] = np.where(pyramid['confirmed'], 'confirmed', 'suspected_not_confirmed')
write(pyramid[['age_group', 'sex', 'status', 'n']], '06_panelE_wave3_age_sex_pyramid.csv')

# ---- 07: panel f, wave 3 positivity by age and sex --------------------------
pos = lin[lin['age_years'].notna() & lin['sex'].isin(['MALE', 'FEMALE'])].copy()
pos['band'] = pd.cut(pos['age_years'], [0, 5, 10, 18, 30, 45, np.inf],
                     labels=['0-4', '5-9', '10-17', '18-29', '30-44', '45+'], right=False)
posit = (pos.groupby(['band', 'sex'], observed=True)
         .agg(suspected=('confirmed', 'size'), confirmed=('confirmed', 'sum')).reset_index())
posit['pct_positive'] = (100 * posit['confirmed'] / posit['suspected']).round(1)
posit['group'] = np.where(posit['band'].astype(str).isin(['0-4', '5-9', '10-17']), 'child (<18y)', 'adult')
write(posit, '07_panelF_wave3_positivity_by_age_sex.csv')

# ---- 08: source tables as transcribed from the sitrep -----------------------
write(ANNUAL, '08_source_ncdc_annual_2017_2023.csv')
write(AGE_BY_YEAR.reset_index(), '09_source_ncdc_confirmed_by_age_year.csv')
write(STATE_BY_YEAR.reset_index(), '10_source_ncdc_confirmed_by_state_year.csv')

readme = """Figure 1 underlying data
========================
00_wave_summary                          one row per wave: the headline numbers quoted in the text
01_panelA_monthly_cases_by_zone          epidemic curve; 2023 is national-only (no state breakdown)
02_panelA_genomes_per_month_by_zone      genome sampling track
03_panelB_state_counts_by_wave           per-state suspected/confirmed and shares, per wave
04_panelC_zone_composition_by_wave       geopolitical-zone composition of confirmed cases
05_panelD_age_composition_by_wave        confirmed cases by NCDC age band, per wave
06_panelE_wave3_age_sex_pyramid          Wave 3 line-list age x sex x result
07_panelF_wave3_positivity_by_age_sex    Wave 3 PCR positivity among suspected cases
08-10_source_ncdc_*                      tables as transcribed from the NCDC sitrep

Sources
  Waves 1-2 (2017-2023): NCDC, 'Update on Mpox (MPX) in Nigeria', Serial 52, Epi-Week 52,
    31 December 2023 - annual indicators (Table 1), confirmed by age band and year (Table 2),
    confirmed by state and year (Table 3), 2023 monthly confirmed (Figure 4).
    Monthly cases by state for 2017-2022 from the NCDC aggregate extract (2017-2022_casescount.csv).
  Wave 3 (2024-2025): NRL national line lists, specimen-collection date, de-duplicated by
    specimen ID; 1,883 suspected and 319 confirmed cases to 31 October 2025.
  Genomes: Nextclade output for Nigerian sequences.

Caveats
  Ascertainment and testing capacity differ between periods; between-wave comparisons are of
  composition, not of absolute incidence. State-level SUSPECTED counts are unavailable for 2023
  and for Waves 1-2 outside the 2017-2022 extract, so Wave 2 suspected totals in 00_wave_summary
  (national, from the sitrep) exceed the sum of the per-state rows in 03_panelB.
"""
open(os.path.join(OUTDIR, 'README.txt'), 'w').write(readme)
print('\nWrote', OUTDIR)
