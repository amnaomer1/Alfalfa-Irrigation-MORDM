"""
Alfalfa Irrigation and Yield Simulation Model

Author: Amna Omer

File Description
----------------
This module implements a daily field-scale alfalfa irrigation and yield
simulation model developed to support Multi-Objective Robust Decision
Making (MORDM) for irrigation management in California's Imperial Valley.

The model:
  1. Estimates reference evapotranspiration (ET0) using FAO-56
     Penman-Monteith or Hamon methods, or accepts precomputed ET0.
  2. Generates daily crop coefficients (Kc) based on growth stages,
     cutting schedules, and temperature-dependent dormancy.
  3. Constructs a multi-year alfalfa cutting calendar, including
     crop establishment and harvest intervals.
  4. Simulates daily soil water balance, crop evapotranspiration,
     and irrigation scheduling under alternative management decisions.
  5. Incorporates seasonal deficit irrigation, soil moisture thresholds,
     irrigation capacity constraints, and survival irrigation.
  6. Estimates cutting-level and annual alfalfa yields using water
     productivity, FAO-33 yield response (Ky), and a normalized
     Mitscherlich yield response function.

Inputs
------
Climate data, soil and crop parameters, cutting-specific yield parameters,
and irrigation management decisions.

Outputs
-------
Daily irrigation, soil moisture, and evapotranspiration; cutting-level
and annual alfalfa yield estimates.

This model serves as the simulation component of the MORDM framework
used to evaluate irrigation water use and crop yield tradeoffs under
uncertainty.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import date

## 1. ETo Estimation

#  1)  Hamon ET0 Function
def ETo_Hamon(T, latitude, doy):

    """
    Daily Hamon ET0.
    Uses mean air temperature, latitude, and day of year to estimate reference ET0.
    """

    T = np.asarray(T, dtype=float)
    doy = np.asarray(doy, dtype=float)

    # Geometry → daylength [hours]
    lat_rad = np.radians(latitude)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)  # solar declination [rad]
    x = -np.tan(lat_rad) * np.tan(decl)
    omega = np.arccos(np.clip(x, -1.0, 1.0))               # sunset hour angle [rad]
    day_length = (24.0 / np.pi) * omega
    Lambda = day_length / 12.0                              # daylight factor [-]

    # Saturation vapor pressure [kPa] (Tetens), piecewise for T < 0 °C
    e_sat = np.where(
        T >= 0.0,
        0.61078 * np.exp((17.27 * T) / (T + 237.3)),
        0.61078 * np.exp((21.875 * T) / (T + 265.5))
    )

    # Saturation vapor density [kg m^-3]
    q_sat = 2.167 * e_sat / (T + 273.15)

    # Hamon ET0 [mm/day]
    ET0 = 330.2 * Lambda * q_sat
    return ET0

#  2)  Penman ET0 Function
def calculate_et0_penman_fao56_full(df: pd.DataFrame, elevation_m: float = 10.0) -> pd.DataFrame:
    
    """
    FAO-56 Penman–Monteith ET0.
    
    Computes daily reference ET0 from temperature, humidity, wind, and radiation.
    Shortwave radiation is approximated from cloud fraction when not observed.
    """

    
    sigma = 4.903e-9           # Stefan–Boltzmann constant [MJ K^-4 m^-2 day^-1]
    P = 101.3                  # Atmospheric pressure [kPa] (assumed constant here)
    gamma = 0.665e-3 * P       # Psychrometric constant [kPa/°C]
    G = 0.0                    # Soil heat flux [MJ m^-2 day^-1], assumed 0 daily

    # Wind speed magnitude
    df["WindSpeed"] = np.sqrt(df["WindU_m_s"]**2 + df["WindV_m_s"]**2)

    # Saturation & actual vapor pressure [kPa]
    df["es"] = 0.6108 * np.exp((17.27 * df["Temperature_C"]) / (df["Temperature_C"] + 237.3))
    df["ea"] = (df["Humidity"] / 100.0) * df["es"]

    # Slope of saturation vapor curve [kPa/°C]
    df["delta"] = (4098 * df["es"]) / ((df["Temperature_C"] + 237.3) ** 2)

    # Vapor pressure deficit [kPa]
    df["VPD"] = df["es"] - df["ea"]

    # Extraterrestrial radiation (Ra) [MJ m^-2 day^-1]
    lat_rad = np.radians(df["latitude"])
    decl = 0.409 * np.sin(2 * np.pi * df["doy"] / 365.0 - 1.39)
    dr = 1 + 0.033 * np.cos(2 * np.pi * df["doy"] / 365.0)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(decl))
    df["Ra"] = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(decl) +
        np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
    )

    # Clear-sky solar radiation [MJ m^-2 day^-1]
    df["Rso"] = (0.75 + 2e-5 * elevation_m) * df["Ra"]

    # Incoming shortwave radiation (estimate from cloud fraction) [MJ m^-2 day^-1]
    df["Rs"] = (1 - df["CloudFraction"]) * df["Rso"]

    # Convert temperature to Kelvin
    df["Temp_K"] = df["Temperature_C"] + 273.15

    # Net outgoing longwave radiation [MJ m^-2 day^-1]
    df["Rnl"] = (
        sigma * df["Temp_K"]**4 *
        (0.34 - 0.14 * np.sqrt(df["ea"])) *
        (1.35 * df["Rs"] / df["Rso"] - 0.35)
    )

    # Net radiation [MJ m^-2 day^-1]
    df["Rn"] = (1 - df["Albedo"]) * df["Rs"] - df["Rnl"]

    # Aerodynamic resistance [s/m], approximate
    rs = 45.0
    df["ra"] = 208.0 / df["WindSpeed"]

    # FAO-56 Penman–Monteith ET0 [mm/day]
    df["ET0"] = (
        (0.408 * df["delta"] * (df["Rn"] - G)) +
        (gamma * (900 / (df["Temperature_C"] + 273.0)) * df["WindSpeed"] * df["VPD"])
    ) / (df["delta"] + gamma * (1 + rs / df["ra"]))

    return df

# 2. Kc Generation

# Helper function: determines dormancy based on 3-day rolling average temperature
def is_dormant_day(current_date, rolling_temp_series, temp_threshold):
    """
    Return True if the day falls in a temperature-based dormancy period
    (using a 3-day rolling mean).
    """

    rolling_temp = rolling_temp_series.get(current_date.normalize(), None)
    return rolling_temp is not None and rolling_temp < temp_threshold

def assign_kc_series(
    cutting_calendar,
    cutting_kc,
    temperature_series,
    stage_lengths_by_cutting_number,
    temp_threshold=6,
    kc_dormant=0.35,
    pre_harvest_buffer_days=3,
    post_cutting_buffer_days=5,
    kc_harvest_day=0.15,
    kc_post_harvest=None,
    kc_pre_harvest_factor=0.6,
    establishment_cutting=0
):
    """
    Build a daily Kc time series across all cuttings using the cutting calendar
    and temperature-based dormancy.
    """

    # Calculate 3-day rolling mean temperature series (to apply dormancy logic)
    temperature_df = pd.Series(temperature_series)
    rolling_temperature_series = temperature_df.rolling(
        window=3,
        center=True,
        min_periods=1
    ).mean().to_dict()

    kc_series = {}

    for _, row in cutting_calendar.iterrows():
        cutting_date = row["Start_Date"]
        cutting_num = row["Cutting"]
        stage_lengths = stage_lengths_by_cutting_number.get(cutting_num, [10, 15, 20, 10])
        total_days = sum(stage_lengths)

        # Unpack Kc values
        Kc_ini, Kc_dev_start, Kc_mid, Kc_late_start, Kc_end = cutting_kc
        post_cut_kc = kc_post_harvest if kc_post_harvest is not None else Kc_ini

        growth_day = 0

        for day in range(total_days + post_cutting_buffer_days + 1):
            current_date = cutting_date + timedelta(days=day)
            is_dormant = is_dormant_day(current_date, rolling_temperature_series, temp_threshold)

            # Main cutting phase
            if day < total_days:
                if is_dormant:
                    kc = kc_dormant
                else:
                    if growth_day < stage_lengths[0]:
                        kc = Kc_ini
                    elif growth_day < stage_lengths[0] + stage_lengths[1]:
                        kc = Kc_dev_start + (Kc_mid - Kc_dev_start) * (
                            (growth_day - stage_lengths[0]) / stage_lengths[1])
                    elif growth_day < stage_lengths[0] + stage_lengths[1] + stage_lengths[2]:
                        kc = Kc_mid
                    else:
                        kc = Kc_late_start + (Kc_end - Kc_late_start) * (
                            (growth_day - (stage_lengths[0] + stage_lengths[1] + stage_lengths[2])) / stage_lengths[3])
                    growth_day += 1
            # Post-harvest period
            elif day <= total_days + post_cutting_buffer_days:
                kc = post_cut_kc
            else:
                kc = kc_dormant

            # Pre-harvest Kc reduction
            if cutting_date + timedelta(days=total_days - pre_harvest_buffer_days) <= current_date < cutting_date + timedelta(days=total_days):
                kc *= kc_pre_harvest_factor

            # Override harvest day Kc
            if current_date == cutting_date + timedelta(days=total_days):
                kc = kc_harvest_day

            kc_series[current_date] = kc

    full_index = pd.date_range(start=cutting_calendar["Start_Date"].min(), end=cutting_calendar["End_Date"].max())
    return pd.Series(kc_series).reindex(full_index).fillna(kc_dormant)

# 3. Cutting Calendar

def build_cutting_calendar(
    start_date,
    years,
    df_climate,
    stage_lengths_by_cutting_number,
    temperature_dormancy_threshold=6,
    min_establishment_days=180,
    max_cuttings_per_year=12,
    post_cutting_buffer_days=5,
    dormancy_threshold=0.5
):
    """
    Generate the multi-year cutting calendar from the planting date,
    using temperature-based dormancy and stage-length constraints.
    """

    cutting_calendar = []
    current_date = pd.Timestamp(start_date)

    df_climate = df_climate.copy()
    df_climate["Dormant"] = (
        df_climate["Temperature_C"]
        .rolling(
            window=3,
            center=True,
            min_periods=1
        )
        .mean()
        < temperature_dormancy_threshold
    )

    #  Establishment cutting
    cut_start = current_date
    cut_end = cut_start + timedelta(days=min_establishment_days)
    cutting_calendar.append({
        "Year": cut_end.year,
        "Cutting": 0,
        "Start_Date": cut_start,
        "End_Date": cut_end,
        "Dormant": False
    })
    current_date = cut_end + timedelta(days=post_cutting_buffer_days)

    # Yearly cutting loop
    for year in years:
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")

        if year_start > df_climate.index[-1]:
            break
        if current_date > year_end:
            continue

        current_date = max(current_date, year_start)
        cut_num = 1

        while current_date <= year_end:
            cut_count_this_year = sum(
                (pd.to_datetime(c["Start_Date"]).year == year and c["Cutting"] != 0)
                for c in cutting_calendar
            )
            if cut_count_this_year >= max_cuttings_per_year:
                break

            if current_date not in df_climate.index or df_climate.loc[current_date, "Dormant"]:
                current_date += timedelta(days=1)
                continue

            stage_lengths = stage_lengths_by_cutting_number.get(cut_num, [5, 10, 10, 5])
            interval = sum(stage_lengths)
            cut_start = current_date
            cut_end = cut_start + timedelta(days=interval)

            if cut_end not in df_climate.index or df_climate.loc[cut_end, "Dormant"]:
                current_date += timedelta(days=1)
                continue

            if cut_end.year != year or cut_end > year_end:
                break

            cut_range = pd.date_range(cut_start, cut_end)
            dormant_days = sum(
                df_climate.loc[d, "Dormant"] for d in cut_range if d in df_climate.index
            )
            if dormant_days / len(cut_range) > dormancy_threshold:
                current_date += timedelta(days=1)
                continue

            cutting_calendar.append({
                "Year": year,
                "Cutting": cut_num,
                "Start_Date": cut_start,
                "End_Date": cut_end,
                "Dormant": False
            })

            current_date = cut_end + timedelta(days=post_cutting_buffer_days)
            cut_num += 1

        current_date = max(current_date, pd.Timestamp(f"{year + 1}-01-01"))

    return pd.DataFrame(cutting_calendar)

# 4. Assign Water productivity for each cutting

def assign_wp(date, cutting_calendar, cutting_param_df):
    """
    Look up the water productivity (WP) for the cutting that contains a given day.
    """
    date = pd.Timestamp(date)
    for _, row in cutting_calendar.iterrows():
        if row["Start_Date"] <= date <= row["End_Date"]:
            match = cutting_param_df[(cutting_param_df["Year"] == row["Year"]) &
                                     (cutting_param_df["Cutting"] == row["Cutting"])]
            return match.iloc[0]["Water_Productivity"] if not match.empty else 17.0
    return 17.0

# 5. Simulate Yield

def compute_yield_with_establishment(
    df_sim,
    cutting_calendar,
    cutting_param_df,
    planting_date,
    yield_min_cutting=0.5,
    min_establishment_days=180,
    k_mitsch_default=3.0
):
    """
    Aggregate daily AETc and PETc to cutting- and annual-scale alfalfa yields,
    using WP, FAO-33 Ky, and a normalized Mitscherlich response.

    Cutting 0 already represents the establishment period and first harvest,
    so establishment AETc and PETc are not added separately.
    """

    def yield_mitscherlich_normalized(r, y_max, k):
        """Normalized Mitscherlich: ensures Y=Ymax at r=1, Y=0 at r=0."""
        r = np.clip(r, 0.0, 1.0)
        k = max(float(k), 1e-6)

        return (
            y_max
            * (1.0 - np.exp(-k * r))
            / (1.0 - np.exp(-k))
        )

    df = df_sim.copy()
    df.index = pd.to_datetime(df.index)

    cutting_calendar = cutting_calendar.copy()
    cutting_calendar["Start_Date"] = pd.to_datetime(
        cutting_calendar["Start_Date"]
    )
    cutting_calendar["End_Date"] = pd.to_datetime(
        cutting_calendar["End_Date"]
    )

    planting_date = pd.Timestamp(planting_date)

    df["Establishment_Cutting"] = False

    first_cut_end_date = (
        planting_date
        + pd.Timedelta(days=min_establishment_days)
    )

    establishment_mask = (
        (df.index >= planting_date)
        & (df.index <= first_cut_end_date)
    )

    df.loc[
        establishment_mask,
        "Establishment_Cutting"
    ] = True

    results = []

    # Loop through each cutting

    for _, row in cutting_calendar.iterrows():

        start = row["Start_Date"]
        end = row["End_Date"]
        year = row["Year"]
        cutting_id = row["Cutting"]

        # Sum AETc and PETc once over this cutting period

        df_cut = df[
            (df.index >= start)
            & (df.index <= end)
        ]

        aet_sum = df_cut["AETc"].sum()
        pet_sum = df_cut["PETc"].sum()

        # Cutting 0 is the first harvest following establishment
        includes_establishment = (cutting_id == 0)

        # Get WP, Ky, and Mitscherlich curvature parameter

        wp_match = cutting_param_df[
            (cutting_param_df["Year"] == year)
            & (cutting_param_df["Cutting"] == cutting_id)
        ]

        wp = (
            wp_match.iloc[0]["Water_Productivity"]
            if not wp_match.empty
            else 17.0
        )

        ky = (
            wp_match.iloc[0]["Ky"]
            if (
                not wp_match.empty
                and "Ky" in wp_match.columns
            )
            else 1.0
        )

        k_mitsch = (
            wp_match.iloc[0]["k_mitsch"]
            if (
                not wp_match.empty
                and "k_mitsch" in wp_match.columns
            )
            else k_mitsch_default
        )

        # 1. WP-based yield

        raw_yield_wp = (
            aet_sum
            * wp
            / 1000.0
        )

        is_skipped = (
            raw_yield_wp
            < yield_min_cutting
        )

        yield_mg_ha_wp = (
            0.0
            if is_skipped
            else raw_yield_wp
        )

        # 2. Ky-based yield

        etc = (
            pet_sum
            if pet_sum > 0
            else 1.0
        )

        eta = aet_sum

        rel_yield_loss = (
            ky
            * (1.0 - eta / etc)
        )

        raw_yield_ky = max(
            0.0,
            (1.0 - rel_yield_loss)
            * (wp * etc / 1000.0)
        )

        yield_mg_ha_ky = (
            0.0
            if raw_yield_ky < yield_min_cutting
            else raw_yield_ky
        )

        # 3. Mitscherlich-based yield

        r = (
            eta / etc
            if etc > 0
            else 0.0
        )

        # Prevent ET ratio outside the physical range
        r = min(
            1.0,
            max(0.0, r)
        )

        # Maximum yield at PETc
        y_max = (
            wp
            * etc
            / 1000.0
        )

        raw_yield_mitsch = (
            yield_mitscherlich_normalized(
                r=r,
                y_max=y_max,
                k=k_mitsch
            )
        )

        yield_mg_ha_mitsch = (
            0.0
            if raw_yield_mitsch < yield_min_cutting
            else raw_yield_mitsch
        )

        # Store cutting results

        results.append({
            "Year": year,
            "Cutting": cutting_id,
            "Start_Date": start,
            "End_Date": end,
            "AETc_mm": aet_sum,
            "PETc_mm": pet_sum,
            "ET_ratio": r,
            "Water_Productivity": wp,
            "Ky": ky,
            "k_mitsch": k_mitsch,
            "Yield_Mg_ha_WP": yield_mg_ha_wp,
            "Yield_Mg_ha_Ky": yield_mg_ha_ky,
            "Yield_Mg_ha_Mitsch": yield_mg_ha_mitsch,
            "Skipped_Cutting": is_skipped,
            "Reason_Skipped": (
                "Yield below threshold"
                if is_skipped
                else ""
            ),
            "Includes_Establishment": includes_establishment
        })

    # Cutting-level results
    df_cuttings = pd.DataFrame(results)

    # Annual totals

    df_yearly = (
        df_cuttings
        .groupby("Year")[
            [
                "Yield_Mg_ha_WP",
                "Yield_Mg_ha_Ky",
                "Yield_Mg_ha_Mitsch"
            ]
        ]
        .sum()
        .reset_index()
    )

    df_yearly.rename(
        columns={
            "Yield_Mg_ha_WP":
                "Total_Yield_Mg_ha_WP",

            "Yield_Mg_ha_Ky":
                "Total_Yield_Mg_ha_Ky",

            "Yield_Mg_ha_Mitsch":
                "Total_Yield_Mg_ha_Mitsch"
        },
        inplace=True
    )

    return (
        df_cuttings,
        df_yearly,
        df
    )

# 6.Assign seasonal deficit irrigation factor
def get_di_factor_from_date(current_date, cutting_calendar, di_levers):
    """
    Return the deficit-irrigation factor based on calendar date.

    Rules:
    - During establishment cutting (cutting 0): use DI_establishment
    - After establishment:
        Dec-Feb -> DI_winter
        Mar-May -> DI_spring
        Jun-Aug -> DI_summer
        Sep-Nov -> DI_fall
    """

    current_date = pd.Timestamp(current_date)

    # Use establishment DI during cutting 0 period
    est_rows = cutting_calendar[cutting_calendar["Cutting"] == 0]
    if not est_rows.empty:
        est_start = pd.Timestamp(est_rows.iloc[0]["Start_Date"])
        est_end = pd.Timestamp(est_rows.iloc[0]["End_Date"])
        if est_start <= current_date <= est_end:
            return float(di_levers["DI_establishment"])

    # Seasonal DI by month
    month = current_date.month

    if month in [12, 1, 2]:
        return float(di_levers["DI_winter"])
    elif month in [3, 4, 5]:
        return float(di_levers["DI_spring"])
    elif month in [6, 7, 8]:
        return float(di_levers["DI_summer"])
    else:
        return float(di_levers["DI_fall"])

# 7.  Irrigation Simulation 

def estimate_infiltration(current_smt, root_depth_mm, soil_params, min_infiltration=3):
    """
    Estimate daily infiltration capacity from current soil moisture
    using a van Genuchten–Mualem conductivity cap.
    """
    
    theta_s = soil_params["theta_s"]
    theta_r = soil_params["theta_r"]
    n = soil_params["n"]
    K_sat = soil_params["K_sat"]

    m = 1 - 1 / n
    theta_wp = soil_params["pwp"]
    theta = theta_wp + current_smt / root_depth_mm
    theta = np.clip(theta, theta_r, theta_s)

    Se = (theta - theta_r) / (theta_s - theta_r)
    Se = np.clip(Se, 0.001, 1.0)

    K = K_sat * (Se ** 0.5) * (1 - (1 - Se ** (1 / m)) ** m) ** 2
    K_limited = max(K, min_infiltration)

    # Limit infiltration by available soil storage
    return float(min(K_limited, max(root_depth_mm * (theta_s - theta), 0)))

def simulate_irrigation(
    df_climate, kc_series, cutting_calendar, 
    start_date, root_depth_mm=1500, awc_mm_per_mm=0.12,
    p_depletion=0.5, initial_smt_fraction=0.7, MAD=0,
    refill_target_fraction=1.0,
    pre_harvest_no_irrigation_days=3,
    post_cutting_buffer_days=5,
    survival_threshold_fraction=0.3,
    use_infiltration_excess=False,
    soil_params=None,
    max_irrigation_per_day=25.0,
    min_infiltration=60,
    di_levers=None
):
    """
    Simulate daily soil moisture, irrigation, and ET using Kc, cutting schedules,
    pre/post-harvest skip rules, and seasonal date-based deficit irrigation.
    """
    # Core water holding properties
    TAW = root_depth_mm * awc_mm_per_mm
    refill_target = refill_target_fraction * TAW
    initial_smt = initial_smt_fraction * TAW
    
    cutting_calendar["Start_Date"] = pd.to_datetime(cutting_calendar["Start_Date"])
    cutting_calendar["End_Date"] = pd.to_datetime(cutting_calendar["End_Date"])

    df = df_climate.copy()
    df = df[df.index >= pd.Timestamp(start_date)]
    df["TAW"] = TAW
    df["Kc"] = kc_series.reindex(df.index).fillna(0.35)
    df["PETc"] = df["Kc"] * df["ET0"]

    # Initialize output columns
    # Initialize float columns
    for col in [
        "Irrigation_mm",
        "Routine_Irrigation_mm",
        "Force_Irrigation_mm",
        "Survival_Irrigation_mm",
        "AETc",
        "Ks",
        "DI_factor",
        "SMT_prev",
        "SMT_final",
        "Peff",
        "Unmet_Irrigation_Request_mm",
        "Irrigation_Needed",
        "Irrigation_Requested",
        "Irrigation_Applied",
        "Infiltration_Capacity"
    ]:
        df[col] = 0.0
    
    # Initialize boolean columns
    df["Skipped_Irrigation"] = False
    df["Irrigation_Limited_By_System"] = False

    # Identify skip dates
    skip_dates = set()
    for _, row in cutting_calendar.iterrows():
        end = pd.Timestamp(row["End_Date"])
        skip_dates.update(pd.date_range(end - timedelta(days=pre_harvest_no_irrigation_days), end))
        skip_dates.update(pd.date_range(end + timedelta(days=1), end + timedelta(days=post_cutting_buffer_days)))

    # Fill missing seasonal DI levers once per simulation
    if di_levers is None:
        di_levers = {}
    else:
        di_levers = di_levers.copy()

    required_keys = ["DI_establishment", "DI_winter", "DI_spring", "DI_summer", "DI_fall"]
    for k in required_keys:
        if k not in di_levers:
            di_levers[k] = 1.0

    smt_yesterday = initial_smt
    unmet_irrigation_request = 0.0
    

    for idx, row in df.iterrows():
        current_date = pd.Timestamp(idx)
        is_skip = current_date in skip_dates
        PETc = row["PETc"]

        # Get DI factor from cutting schedule or Dates

        if is_skip:
            di_factor = 0.0
        else:
            di_factor = get_di_factor_from_date(current_date, cutting_calendar, di_levers)
    
        # Infiltration and effective precipitation
        
        # Physical storage remaining in the root zone
        physical_storage_available = max(
            TAW - smt_yesterday,
            0.0
        )
        
        if use_infiltration_excess:
        
            infiltration_capacity = estimate_infiltration(
                smt_yesterday,
                root_depth_mm,
                soil_params,
                min_infiltration=min_infiltration
            )
        
        else:
        
            # Without the infiltration-excess formulation,
            # infiltration is limited by remaining root-zone storage
            infiltration_capacity = physical_storage_available
        
        
        # Prevent negative precipitation values
        precip_today = max(
            float(row["Precip_mm_day"]),
            0.0
        )
        
        # Effective precipitation cannot exceed infiltration capacity
        peff = min(
            precip_today,
            infiltration_capacity
        )
        
        smt_rain = smt_yesterday + peff

        # Calculate Readily Available Water
        RAW = p_depletion * TAW              # Readily Available Water [mm]

        # If SMT is remaining available water, stress starts when SMT < (TAW - RAW)
        stress_threshold = TAW - RAW

        if smt_yesterday >= stress_threshold:
            Ks = 1.0
        else:
            Ks = smt_yesterday / max(stress_threshold, 1e-9)
            Ks = max(0.0, min(1.0, Ks))

        # Decide if irrigation is needed
        irrigation_needed = max(0.0, refill_target - smt_rain)

        # Emergency override: allow irrigation if Ks == 0 even during skip period
        force_irrigation = (Ks == 0 and is_skip)

        # Decide if irrigation is allowed (based on SMT dropping below trigger threshold)
        trigger_threshold =  TAW * (1-MAD)
        allow_irrigation_today = smt_yesterday <= trigger_threshold or force_irrigation
        
        # Determine irrigation request
        if not is_skip and allow_irrigation_today:
            irrigation_requested = irrigation_needed * di_factor
        elif force_irrigation:
            irrigation_requested = irrigation_needed  # Apply full amount, no DI
        else:
            irrigation_requested = 0.0

        # Remaining infiltration capacity after precipitation
        available_for_irrigation = max(
            infiltration_capacity - peff,
            0.0
        )
        
        irrigation = min(
            irrigation_requested,
            available_for_irrigation,
            max_irrigation_per_day
        )
        
        # Save irrigation applied before survival top-up
        normal_irrigation = irrigation
        
        # Track source of normal/pre-survival irrigation
        
        if force_irrigation:
            force_irrigation_applied = irrigation
            routine_irrigation_applied = 0.0
        else:
            force_irrigation_applied = 0.0
            routine_irrigation_applied = irrigation
        
        excess_irrigation = max(
            0.0,
            irrigation_requested - irrigation
        )
        
        unmet_irrigation_request = excess_irrigation

        AETc = Ks * PETc
        smt_final = smt_yesterday + peff + irrigation - AETc
        smt_final = min(smt_final, TAW)
        smt_final = max(0, smt_final)


        survival_threshold = survival_threshold_fraction * TAW
        
        # Survival irrigation
        survival_irrigation = 0.0
        
        if smt_final < survival_threshold:
        
            # Water needed to reach the survival threshold
            survival_needed = survival_threshold - smt_final
        
            # Remaining daily system-delivery capacity
            remaining_system_capacity = max(
                max_irrigation_per_day - normal_irrigation,
                0.0
            )
        
            # Remaining infiltration capacity after precipitation
            # and normal/force irrigation
            remaining_infiltration_capacity = max(
                infiltration_capacity - peff - normal_irrigation,
                0.0
            )
        
            # Survival irrigation is prioritized, but it cannot
            # exceed the remaining physical/system capacity
            survival_irrigation = min(
                survival_needed,
                remaining_system_capacity,
                remaining_infiltration_capacity
            )
        
            irrigation += survival_irrigation
            smt_final += survival_irrigation
        

        # Track results
        df.at[idx, "Peff"] = peff
        df.at[idx, "Ks"] = Ks
        df.at[idx, "DI_factor"] = di_factor
        
        # Total irrigation
        df.at[idx, "Irrigation_mm"] = irrigation
        
        # Irrigation components
        df.at[idx, "Routine_Irrigation_mm"] = routine_irrigation_applied
        df.at[idx, "Force_Irrigation_mm"] = force_irrigation_applied
        df.at[idx, "Survival_Irrigation_mm"] = survival_irrigation
        
        df.at[idx, "AETc"] = AETc
        df.at[idx, "SMT_prev"] = smt_yesterday
        df.at[idx, "SMT_final"] = smt_final
        df.at[idx, "Skipped_Irrigation"] = is_skip
        df.at[idx, "Unmet_Irrigation_Request_mm"] = unmet_irrigation_request
        df.at[idx, "Irrigation_Needed"] = irrigation_needed
        df.at[idx, "Irrigation_Requested"] = irrigation_requested
        df.at[idx, "Irrigation_Applied"] = irrigation
        df.at[idx, "Infiltration_Capacity"] = infiltration_capacity
                
        df.at[idx, "Irrigation_Limited_By_System"] = (
            normal_irrigation < irrigation_requested
            and (not is_skip or force_irrigation)
        )

        smt_yesterday = smt_final

    return df

#  Wrapper for full simulation with all user parameters

def full_alfalfa_simulation(
    climate_csv_path,
    output_csv_path,
    cutting_yield_output_csv_path,
    yearly_yield_output_csv_path,
    start_date,
    years,
    cutting_kc,
    root_depth_mm,
    awc_mm_per_mm,
    p_depletion,
    MAD,
    initial_smt_fraction,
    cutting_param_csv_path=None,
    cutting_param_df=None,
    climate_df=None,
    et0_method="penman",
    refill_target_fraction=1.0,  
    yield_min_cutting=0.5,
    k_mitsch_default=3,   
    temperature_dormancy_threshold=6,
    pre_harvest_no_irrigation_days=3,
    post_cutting_buffer_days=5,
    min_establishment_days=180,
    stage_lengths_by_cutting_number=None,
    kc_dormant=0.35,
    kc_harvest_day=0.15,
    kc_post_harvest=0.25,
    kc_pre_harvest_factor=0.6,
    establishment_cutting=0,  
    survival_threshold_fraction=0.3,
    use_infiltration_excess=False,
    soil_params=None,
    max_irrigation_per_day=25.0,
    min_infiltration=60,
    di_levers=None
):
    """
    Run the alfalfa simulation and write daily, cutting, and yearly outputs to CSV.
    """

    if climate_df is not None:
        df_climate = climate_df.copy()
    
        if isinstance(df_climate, pd.DataFrame):
            if 'Date' in df_climate.columns:
                df_climate.set_index('Date', inplace=True)
                df_climate.index = pd.to_datetime(df_climate.index)
            elif df_climate.index.name != 'Date':
                raise ValueError("Provided climate_df must have a 'Date' column or DateTimeIndex named 'Date'")
        else:
            raise TypeError("climate_df must be a pandas DataFrame")
    
    elif climate_csv_path is not None:
        df_climate = pd.read_csv(climate_csv_path, parse_dates=["Date"])
        df_climate.set_index("Date", inplace=True)
        df_climate.index = pd.to_datetime(df_climate.index)

    
    else:
        raise ValueError("Must provide either climate_df or climate_csv_path")

    if cutting_param_df is None:
        if cutting_param_csv_path is not None:
            cutting_param_df = pd.read_csv(cutting_param_csv_path)
        else:
            raise ValueError("Either cutting_param_df or cutting_param_csv_path must be provided.")
        
    if "k_mitsch" not in cutting_param_df.columns:
        cutting_param_df["k_mitsch"] = k_mitsch_default
    else:
        cutting_param_df["k_mitsch"] = cutting_param_df["k_mitsch"].fillna(k_mitsch_default)


    df_climate["doy"] = df_climate.index.dayofyear
    
    if et0_method.lower() == "hamon":
        df_climate["ET0"] = ETo_Hamon(
            df_climate["Temperature_C"].values,
            df_climate["latitude"].values[0],
            df_climate["doy"].values
        )
    elif et0_method.lower() == "penman":
        df_climate = calculate_et0_penman_fao56_full(df_climate, elevation_m=10)
    elif et0_method.lower() == "precomputed":
        # ET0 is already assumed to be in the climate data
        if "ET0" not in df_climate.columns:
            raise ValueError("ET0 column not found in climate data for precomputed method.")
        pass  # No calculation needed
    else:
        raise ValueError(f"Invalid ET0 method: {et0_method}. Choose 'hamon', 'penman', or 'precomputed'.")

    start_date = pd.Timestamp(start_date)
    # Build cutting calendar
    cutting_calendar = build_cutting_calendar(
        start_date=start_date,
        years=years,
        df_climate=df_climate,
        stage_lengths_by_cutting_number=stage_lengths_by_cutting_number,
        min_establishment_days=min_establishment_days,
        temperature_dormancy_threshold=temperature_dormancy_threshold,
        max_cuttings_per_year=12,
        post_cutting_buffer_days=post_cutting_buffer_days,
        dormancy_threshold=0.5
    )

    # Temperature time series for Kc assignment
    temperature_series = df_climate["Temperature_C"].to_dict()

    # Generate Kc time series
    kc_series = assign_kc_series(
        cutting_calendar=cutting_calendar,
        cutting_kc=cutting_kc,
        temperature_series=temperature_series,
        stage_lengths_by_cutting_number=stage_lengths_by_cutting_number,
        temp_threshold=temperature_dormancy_threshold,
        kc_dormant=kc_dormant,
        pre_harvest_buffer_days=pre_harvest_no_irrigation_days,
        post_cutting_buffer_days=post_cutting_buffer_days,
        kc_harvest_day=kc_harvest_day,
        kc_post_harvest=kc_post_harvest,
        kc_pre_harvest_factor=kc_pre_harvest_factor,
        establishment_cutting=0
    )

    # Simulate irrigation and water balance
    df_result = simulate_irrigation(
        df_climate=df_climate,
        kc_series=kc_series,
        cutting_calendar=cutting_calendar,
        start_date=start_date,
        root_depth_mm=root_depth_mm,
        awc_mm_per_mm=awc_mm_per_mm,
        p_depletion=p_depletion,
        MAD=MAD,
        initial_smt_fraction=initial_smt_fraction,
        refill_target_fraction=refill_target_fraction,
        pre_harvest_no_irrigation_days=pre_harvest_no_irrigation_days,
        post_cutting_buffer_days=post_cutting_buffer_days,
        survival_threshold_fraction=survival_threshold_fraction,
        use_infiltration_excess=use_infiltration_excess,
        soil_params=soil_params,
        max_irrigation_per_day=max_irrigation_per_day,
        min_infiltration=min_infiltration,
        di_levers=di_levers
    )

    if "Daily_Yield_Mg_ha" not in df_result.columns:
        df_result["Daily_Yield_Mg_ha"] = 0.0

    # Compute yields and assign establishment yield
    df_cuttings, df_yearly, df_result = compute_yield_with_establishment(
        df_sim=df_result,
        cutting_calendar=cutting_calendar,
        cutting_param_df=cutting_param_df,
        planting_date=start_date,
        yield_min_cutting=yield_min_cutting,
        min_establishment_days=min_establishment_days,
        k_mitsch_default=k_mitsch_default   
    )

    # Export outputs
    if output_csv_path is not None:
        df_result.to_csv(output_csv_path)

    if cutting_yield_output_csv_path is not None:
        df_cuttings.to_csv(cutting_yield_output_csv_path, index=False)

    if yearly_yield_output_csv_path is not None:
        df_yearly.to_csv(yearly_yield_output_csv_path, index=False)
  

    return df_result, df_cuttings, df_yearly
