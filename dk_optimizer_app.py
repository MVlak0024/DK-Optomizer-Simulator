"""
DraftKings Lineup Optimizer & Simulator - Phone-Friendly Edition
Supports NFL Classic (primary) + NBA/MLB/NHL.
Uses pydfs-lineup-optimizer + Monte Carlo simulator.
Optimized for mobile browsers.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt

from pydfs_lineup_optimizer import get_optimizer, Site, Sport, Player, TeamStack
from pydfs_lineup_optimizer.exceptions import LineupOptimizerException

# --------------------------
# Page config - mobile friendly
# --------------------------
st.set_page_config(
    page_title="DK Optimizer",
    page_icon="🏈",
    layout="centered",          # better on phones than "wide"
    initial_sidebar_state="collapsed",  # start collapsed on mobile
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "DraftKings Lineup Optimizer & Simulator – phone-friendly edition"
    }
)

# Custom CSS for touch-friendly mobile UI
st.markdown("""
<style>
    /* Larger touch targets */
    .stButton > button {
        width: 100%;
        min-height: 3rem;
        font-size: 1.1rem;
        border-radius: 12px;
        font-weight: 600;
    }
    /* Bigger metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
    /* Tighter padding on mobile */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 700px;
    }
    /* Better selectboxes and inputs */
    .stSelectbox, .stMultiSelect, .stNumberInput, .stSlider {
        margin-bottom: 0.6rem;
    }
    /* File uploader */
    [data-testid="stFileUploader"] {
        padding: 0.5rem;
    }
    /* Expander headers larger */
    .streamlit-expanderHeader {
        font-size: 1.05rem;
        font-weight: 600;
    }
    /* Dataframes scroll horizontally if needed */
    .stDataFrame {
        font-size: 0.85rem;
    }
    /* Sidebar tighter */
    section[data-testid="stSidebar"] {
        width: 280px !important;
    }
    /* Success/error boxes */
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Helpers
# --------------------------

SPORT_MAP = {
    "NFL Classic": (Site.DRAFTKINGS, Sport.FOOTBALL),
    "NBA Classic": (Site.DRAFTKINGS, Sport.BASKETBALL),
    "MLB Classic": (Site.DRAFTKINGS, Sport.BASEBALL),
    "NHL Classic": (Site.DRAFTKINGS, Sport.HOCKEY),
}

NFL_POSITIONS = ["QB", "RB", "WR", "TE", "DST"]


def load_dk_csv(uploaded_file) -> pd.DataFrame:
    """Load DraftKings salaries CSV and normalize columns."""
    df = pd.read_csv(uploaded_file)
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("name", "player"):
            col_map[c] = "Name"
        elif cl in ("id", "player id", "playerid"):
            col_map[c] = "ID"
        elif "position" in cl and "roster" not in cl:
            col_map[c] = "Position"
        elif cl == "salary":
            col_map[c] = "Salary"
        elif "team" in cl:
            col_map[c] = "Team"
        elif any(x in cl for x in ("avgpoints", "fppg", "projection", "proj", "points")):
            col_map[c] = "Projection"
        elif "game info" in cl or "gameinfo" in cl:
            col_map[c] = "GameInfo"
    df = df.rename(columns=col_map)

    required = ["Name", "Position", "Salary"]
    for r in required:
        if r not in df.columns:
            st.error(f"Missing column: **{r}**. Found: {list(df.columns)}")
            return pd.DataFrame()

    if "ID" not in df.columns:
        df["ID"] = range(1, len(df) + 1)
    if "Team" not in df.columns:
        df["Team"] = "UNK"
    if "Projection" not in df.columns:
        df["Projection"] = 0.0
        st.warning("No projection column found – set to 0. Upload projections or edit below.")

    df["Position"] = df["Position"].astype(str).str.upper().str.strip()
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int)
    df["Projection"] = pd.to_numeric(df["Projection"], errors="coerce").fillna(0.0)
    df["Name"] = df["Name"].astype(str).str.strip()
    return df


def merge_projections(players_df: pd.DataFrame, proj_file) -> pd.DataFrame:
    """Merge external projections by name."""
    proj = pd.read_csv(proj_file)
    name_col = pts_col = None
    for c in proj.columns:
        cl = c.lower()
        if name_col is None and ("name" in cl or "player" in cl):
            name_col = c
        if pts_col is None and any(x in cl for x in ("proj", "fpts", "points", "pts", "fp")):
            pts_col = c
    if name_col is None or pts_col is None:
        st.error("Projections CSV needs a **Name** column and a **points/projection** column.")
        return players_df

    proj = proj[[name_col, pts_col]].copy()
    proj.columns = ["Name", "Projection"]
    proj["Name"] = proj["Name"].astype(str).str.strip()
    players_df = players_df.drop(columns=["Projection"], errors="ignore")
    merged = players_df.merge(proj, on="Name", how="left")

    # Simple fuzzy match for misses
    missing = merged["Projection"].isna()
    if missing.any():
        from difflib import get_close_matches
        name_to_proj = dict(zip(proj["Name"], proj["Projection"]))
        for idx in merged[missing].index:
            matches = get_close_matches(merged.loc[idx, "Name"], list(name_to_proj.keys()), n=1, cutoff=0.82)
            if matches:
                merged.loc[idx, "Projection"] = name_to_proj[matches[0]]
    merged["Projection"] = merged["Projection"].fillna(0.0)
    return merged


def create_players_from_df(df: pd.DataFrame) -> List[Player]:
    """Convert DataFrame to pydfs Player objects."""
    players = []
    for _, row in df.iterrows():
        positions = [p.strip() for p in str(row["Position"]).replace("/", " ").split() if p.strip()]
        if not positions:
            positions = ["FLEX"]
        full_name = str(row["Name"]).strip()
        parts = full_name.split()
        first_name = parts[0] if parts else full_name
        last_name = " ".join(parts[1:]) if len(parts) > 1 else ""
        try:
            p = Player(
                player_id=str(row.get("ID", row.name)),
                first_name=first_name,
                last_name=last_name,
                positions=positions,
                team=str(row.get("Team", "UNK")),
                salary=float(row["Salary"]),
                fppg=float(row["Projection"]),
            )
            players.append(p)
        except Exception:
            continue
    return players


def lineup_to_dict(lineup) -> Dict:
    """Convert pydfs Lineup to serializable dict."""
    players = []
    for p in lineup.players:
        name = getattr(p, "full_name", None) or f"{getattr(p, 'first_name', '')} {getattr(p, 'last_name', '')}".strip()
        players.append({
            "name": name,
            "position": getattr(p, "lineup_position", None) or (p.positions[0] if p.positions else ""),
            "team": p.team,
            "salary": p.salary,
            "proj": p.fppg,
            "id": p.id,
        })
    return {
        "players": players,
        "salary": lineup.salary_costs,
        "proj": lineup.fantasy_points_projection,
    }


def simulate_lineups(
    lineups: List[Dict],
    player_stats: Dict[str, Tuple[float, float]],
    n_sims: int = 4000,
    cash_line_pct: float = 0.20,
) -> pd.DataFrame:
    """Monte Carlo simulation of lineup scores."""
    results = []
    for i, lu in enumerate(lineups):
        scores = np.zeros(n_sims)
        for p in lu["players"]:
            name = p["name"]
            mean, std = player_stats.get(name, (p["proj"], max(p["proj"] * 0.35, 3.0)))
            scores += np.random.normal(mean, std, n_sims)
        scores = np.maximum(scores, 0)
        mean_score = scores.mean()
        std_score = scores.std()
        p90 = np.percentile(scores, 90)
        p10 = np.percentile(scores, 10)

        # Rough cash-rate estimate
        field_mean = mean_score * 0.96
        field_std = std_score * 1.1
        cash_threshold = np.percentile(np.random.normal(field_mean, field_std, 2500), 100 * (1 - cash_line_pct))
        cash_rate = (scores >= cash_threshold).mean()

        results.append({
            "Lineup #": i + 1,
            "Mean": round(mean_score, 1),
            "Std": round(std_score, 1),
            "Floor": round(p10, 1),
            "Ceil": round(p90, 1),
            "Cash %": round(cash_rate * 100, 1),
            "Raw Scores": scores,
        })
    return pd.DataFrame(results)


def make_demo_data() -> pd.DataFrame:
    """Synthetic NFL slate large enough for valid lineups."""
    return pd.DataFrame({
        "Name": [
            "Patrick Mahomes", "Josh Allen", "Lamar Jackson", "Jalen Hurts", "Joe Burrow", "Dak Prescott",
            "Christian McCaffrey", "Bijan Robinson", "Saquon Barkley", "Derrick Henry", "Jahmyr Gibbs", "Breece Hall",
            "Kyren Williams", "Alvin Kamara", "James Cook", "David Montgomery",
            "CeeDee Lamb", "Tyreek Hill", "Ja'Marr Chase", "Amon-Ra St. Brown", "Justin Jefferson", "A.J. Brown",
            "DK Metcalf", "Mike Evans", "Davante Adams", "Stefon Diggs", "Chris Olave", "Garrett Wilson",
            "DeVonta Smith", "Puka Nacua", "Nico Collins", "Brandon Aiyuk",
            "Travis Kelce", "George Kittle", "Mark Andrews", "T.J. Hockenson", "Sam LaPorta", "Evan Engram",
            "Chiefs", "Bills", "Ravens", "Eagles", "49ers", "Lions", "Bengals", "Cowboys"
        ],
        "Position": (["QB"] * 6 + ["RB"] * 10 + ["WR"] * 16 + ["TE"] * 6 + ["DST"] * 8),
        "Team": [
            "KC", "BUF", "BAL", "PHI", "CIN", "DAL",
            "SF", "ATL", "PHI", "BAL", "DET", "NYJ", "LAR", "NO", "BUF", "DET",
            "DAL", "MIA", "CIN", "DET", "MIN", "PHI", "SEA", "TB", "LV", "HOU", "NO", "NYJ", "PHI", "LAR", "HOU", "SF",
            "KC", "SF", "BAL", "MIN", "DET", "JAX",
            "KC", "BUF", "BAL", "PHI", "SF", "DET", "CIN", "DAL"
        ],
        "Salary": [
            7800, 7600, 7400, 7200, 7000, 6800,
            9000, 7800, 7600, 7000, 6800, 6500, 6400, 6200, 6000, 5800,
            8200, 8000, 7900, 7500, 7700, 7400, 6200, 6000, 5900, 5800, 5600, 5500, 5400, 5300, 5200, 5100,
            6000, 5500, 5200, 4800, 4700, 4500,
            3200, 3100, 3000, 2900, 2800, 2700, 2600, 2500
        ],
        "Projection": [
            24.5, 23.8, 22.1, 21.5, 20.8, 19.5,
            22.0, 18.5, 17.8, 16.2, 15.5, 14.8, 14.2, 13.8, 13.5, 12.9,
            19.5, 18.2, 17.9, 16.8, 17.5, 16.2, 13.5, 12.8, 12.5, 12.2, 11.8, 11.5, 11.2, 11.0, 10.8, 10.5,
            14.5, 12.0, 11.5, 10.2, 9.8, 9.2,
            8.5, 8.0, 7.8, 7.5, 7.2, 7.0, 6.8, 6.5
        ],
        "ID": list(range(1, 47))
    })


# --------------------------
# App UI
# --------------------------

st.title("🏈 DK Optimizer")
st.caption("Lineup optimizer + simulator • Phone-friendly")

# ---- Sidebar (collapsed by default on mobile) ----
with st.sidebar:
    st.header("⚙️ Settings")
    sport_choice = st.selectbox("Sport", list(SPORT_MAP.keys()), index=0)
    site, sport = SPORT_MAP[sport_choice]

    n_lineups = st.number_input("Lineups to generate", 1, 100, 15, 1)
    max_exposure = st.slider("Max player exposure %", 10, 100, 40) / 100.0
    min_salary = st.number_input("Min salary used", 0, 50000, 49000, 500)

    st.divider()
    st.subheader("NFL Stacking")
    qb_stack = st.checkbox("QB + pass-catcher stack", value=True)
    stack_size = st.selectbox("Pass-catchers with QB", [1, 2], index=0)

    st.divider()
    st.subheader("Simulation")
    n_sims = st.number_input("Sims per lineup", 1000, 15000, 4000, 500)
    std_pct = st.slider("Player std (% of proj)", 20, 50, 35) / 100.0

    st.divider()
    st.caption("Tip: On phone, tap ☰ (top left) to open settings.")

# ---- Main flow ----
st.header("1. Load Players")

# Demo button first (very useful on phone)
if st.button("📥 Load Demo NFL Data", use_container_width=True):
    st.session_state["players_df"] = make_demo_data()
    st.rerun()

salaries_file = st.file_uploader("DraftKings Salaries CSV", type=["csv"], help="Export from any DK contest lobby")
proj_file = st.file_uploader("Optional Projections CSV", type=["csv"], help="Columns: Name + projected points")

players_df = st.session_state.get("players_df")

if salaries_file is not None:
    players_df = load_dk_csv(salaries_file)
    if not players_df.empty and proj_file is not None:
        players_df = merge_projections(players_df, proj_file)
    if not players_df.empty:
        st.session_state["players_df"] = players_df

if players_df is not None and not players_df.empty:
    st.success(f"**{len(players_df)}** players loaded  •  Salary ${players_df['Salary'].min():,} – ${players_df['Salary'].max():,}")

    with st.expander("👀 Preview top players", expanded=False):
        preview = players_df[["Name", "Position", "Team", "Salary", "Projection"]].sort_values("Projection", ascending=False).head(30)
        st.dataframe(preview, use_container_width=True, height=280)

    # ---- Player controls ----
    st.header("2. Controls")
    with st.expander("Lock / Exclude players", expanded=False):
        lock_players = st.multiselect("🔒 Lock (must include)", players_df["Name"].tolist(), default=[])
        exclude_players = st.multiselect("🚫 Exclude", players_df["Name"].tolist(), default=[])

    with st.expander("✏️ Edit top projections", expanded=False):
        edit_df = players_df[["Name", "Position", "Team", "Salary", "Projection"]].sort_values("Projection", ascending=False).head(40).copy()
        edited = st.data_editor(
            edit_df,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "Projection": st.column_config.NumberColumn(min_value=0.0, max_value=50.0, step=0.1, format="%.1f"),
                "Salary": st.column_config.NumberColumn(format="$%d"),
            },
            key="proj_editor",
            height=320,
        )
        for _, row in edited.iterrows():
            mask = players_df["Name"] == row["Name"]
            players_df.loc[mask, "Projection"] = row["Projection"]
        st.session_state["players_df"] = players_df

    # ---- Optimize ----
    st.header("3. Optimize")
    if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
        with st.spinner("Optimizing…"):
            try:
                optimizer = get_optimizer(site, sport)
                pool_df = players_df[~players_df["Name"].isin(exclude_players)].copy()
                players = create_players_from_df(pool_df)
                optimizer.player_pool.load_players(players)

                # Locks
                for name in lock_players:
                    matches = [p for p in optimizer.player_pool.get_players()
                               if (getattr(p, "full_name", "") == name or
                                   f"{p.first_name} {p.last_name}".strip() == name)]
                    if matches:
                        try:
                            optimizer.add_player_to_lineup(matches[0])
                        except Exception:
                            pass

                # Exposure
                for p in optimizer.player_pool.get_players():
                    p.max_exposure = max_exposure

                if min_salary > 0:
                    optimizer.set_min_salary_cap(min_salary)

                # Stacking
                if "NFL" in sport_choice and qb_stack:
                    optimizer.add_stack(
                        TeamStack(stack_size + 1, for_positions=["QB", "WR", "TE"], max_exposure=1.0)
                    )

                lineups_gen = optimizer.optimize(n=n_lineups)
                lineups = [lineup_to_dict(lu) for lu in lineups_gen]
                st.session_state["lineups"] = lineups
                st.success(f"Generated **{len(lineups)}** lineups")
            except LineupOptimizerException as e:
                st.error(f"Optimizer error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

    # ---- Show lineups ----
    if "lineups" in st.session_state and st.session_state["lineups"]:
        lineups = st.session_state["lineups"]
        st.subheader(f"Lineups ({len(lineups)})")

        # Compact summary
        summary_data = []
        for i, lu in enumerate(lineups):
            names = [p["name"].split()[-1] for p in lu["players"]]  # last names for space
            summary_data.append({
                "#": i + 1,
                "Proj": round(lu["proj"], 1),
                "Sal": lu["salary"],
                "Core": " · ".join(names[:4]) + ("…" if len(names) > 4 else ""),
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, height=260)

        # Inspect one
        selected = st.selectbox("Inspect lineup", range(1, len(lineups) + 1), format_func=lambda x: f"#{x}")
        lu = lineups[selected - 1]

        col1, col2 = st.columns(2)
        col1.metric("Projected Pts", f"{lu['proj']:.1f}")
        col2.metric("Salary", f"${lu['salary']:,}")

        detail = pd.DataFrame(lu["players"])[["position", "name", "team", "salary", "proj"]]
        detail.columns = ["Pos", "Player", "Team", "Sal", "Proj"]
        st.dataframe(detail, use_container_width=True, hide_index=True)

        # Export
        export_rows = []
        for i, lu in enumerate(lineups):
            row = {"Entry": i + 1}
            for p in lu["players"]:
                pos = p["position"] or "FLEX"
                row[pos] = p["name"]
            row["Salary"] = lu["salary"]
            row["Proj"] = round(lu["proj"], 1)
            export_rows.append(row)
        export_df = pd.DataFrame(export_rows)
        st.download_button(
            "⬇️ Download Lineups CSV",
            data=export_df.to_csv(index=False),
            file_name="dk_lineups.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ---- Simulator ----
        st.header("4. Simulate")
        st.caption(f"Monte Carlo • {n_sims:,} sims • std ≈ {std_pct*100:.0f}% of projection")

        if st.button("🎲 Run Simulation", use_container_width=True):
            player_stats = {}
            for _, row in players_df.iterrows():
                mean = float(row["Projection"])
                std = max(mean * std_pct, 2.5)
                player_stats[row["Name"]] = (mean, std)

            with st.spinner(f"Running {n_sims:,} simulations…"):
                sim_df = simulate_lineups(lineups, player_stats, n_sims=n_sims)
                st.session_state["sim_df"] = sim_df

        if "sim_df" in st.session_state:
            sim_df = st.session_state["sim_df"]
            display = sim_df.drop(columns=["Raw Scores"]).copy()
            st.dataframe(display, use_container_width=True, height=280)

            # Quick bests
            best_mean_idx = sim_df["Mean"].idxmax()
            best_ceil_idx = sim_df["Ceil"].idxmax()
            st.info(f"Highest mean: **#{int(sim_df.loc[best_mean_idx, 'Lineup #'])}** ({sim_df.loc[best_mean_idx, 'Mean']} pts)")
            st.info(f"Highest ceiling: **#{int(sim_df.loc[best_ceil_idx, 'Lineup #'])}** ({sim_df.loc[best_ceil_idx, 'Ceil']} pts)")

            # Compact histogram of first 4
            with st.expander("📊 Score distributions (first 4 lineups)"):
                fig, ax = plt.subplots(figsize=(7, 3.5))
                for i in range(min(4, len(sim_df))):
                    scores = sim_df.iloc[i]["Raw Scores"]
                    ax.hist(scores, bins=35, alpha=0.5, label=f"#{i+1}")
                ax.set_xlabel("Fantasy Points")
                ax.set_ylabel("Count")
                ax.legend(fontsize=8)
                ax.set_title("Simulated Scores")
                st.pyplot(fig, use_container_width=True)

else:
    st.info("Upload a DraftKings salaries CSV **or** tap the demo button above to try the app.")

st.divider()
st.caption("Not affiliated with DraftKings. Always verify lineups & rules on the official site.")
