import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from logic import build_and_train, latest_team_stats_from_df


class PredictorApp:
    def __init__(self, root, artifacts):
        self.root = root
        self.artifacts = artifacts
        self.model = None
        self.le_home = None
        self.le_away = None
        self.features = None
        self.df_raw = None

        root.title("⚽ Football Match Predictor (Premier League 18/19 demo)")
        root.geometry("800x500")
        root.configure(bg="#222831")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background="#222831")
        style.configure('TLabel', background="#222831", foreground="#eeeeee", font=("Segoe UI", 14, "bold"))
        style.configure('TButton', font=("Segoe UI", 13, "bold"), padding=8, background="#00adb5", foreground="#222831")
        style.map('TButton', background=[('active', '#393e46')], foreground=[('active', '#00adb5')])
        style.configure('TCombobox', font=("Segoe UI", 13))

        frm = ttk.Frame(root, padding=24, style='TFrame')
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Home Team:", style='TLabel').grid(column=0, row=0, sticky=tk.W, pady=16, padx=8)
        ttk.Label(frm, text="Away Team:", style='TLabel').grid(column=0, row=1, sticky=tk.W, pady=16, padx=8)

        # League selection
        self.league_var = tk.StringVar()
        leagues = [
            ("PL", "English Premier League"),
            ("FL", "French Ligue 1"),
            ("GL", "German Bundesliga"),
            ("IL", "Italian Serie A"),
            ("SL", "Spanish La Liga")
        ]
        self.league_files = {
            "PL": os.path.join("dataset", "season-1819-PL.csv"),
            "FL": os.path.join("dataset", "season-1819-FL.csv"),
            "GL": os.path.join("dataset", "season-1819-GL.csv"),
            "IL": os.path.join("dataset", "season-1819-IL.csv"),
            "SL": os.path.join("dataset", "season-1819-SL.csv")
        }

        # Accuracy label
        self.accuracy_var = tk.StringVar(value="Test accuracy: N/A")
        self.accuracy_label = ttk.Label(frm, textvariable=self.accuracy_var, style='TLabel', font=("Segoe UI", 12, "italic"))
        self.accuracy_label.grid(column=0, row=3, columnspan=2, sticky=tk.W, padx=8, pady=(0, 8))
        ttk.Label(frm, text="League:", style='TLabel').grid(column=0, row=0, sticky=tk.W, pady=16, padx=8)
        self.league_cb = ttk.Combobox(frm, textvariable=self.league_var, values=[f"{code}: {name}" for code, name in leagues], state='readonly', width=32, font=("Segoe UI", 13))
        self.league_cb.grid(column=1, row=0, padx=12, pady=16, sticky=tk.W)
        self.league_cb.bind("<<ComboboxSelected>>", self.on_league_change)

        # Home/Away team selectors (start empty)
        self.home_var = tk.StringVar()
        self.away_var = tk.StringVar()
        self.home_cb = ttk.Combobox(frm, textvariable=self.home_var, values=[], state='readonly', width=32, font=("Segoe UI", 13))
        self.away_cb = ttk.Combobox(frm, textvariable=self.away_var, values=[], state='readonly', width=32, font=("Segoe UI", 13))
        ttk.Label(frm, text="Home Team:", style='TLabel').grid(column=0, row=1, sticky=tk.W, pady=16, padx=8)
        ttk.Label(frm, text="Away Team:", style='TLabel').grid(column=0, row=2, sticky=tk.W, pady=16, padx=8)
        self.home_cb.grid(column=1, row=1, padx=12, pady=16, sticky=tk.W)
        self.away_cb.grid(column=1, row=2, padx=12, pady=16, sticky=tk.W)

        btn_predict = ttk.Button(frm, text="Predict", command=self.predict, style='TButton')
        btn_predict.grid(column=0, row=4, columnspan=2, pady=24, padx=8, sticky=tk.EW)

        self.result_text = tk.Text(frm, height=12, width=80, font=("Consolas", 12), bg="#393e46", fg="#00adb5", wrap=tk.WORD, borderwidth=2, relief="groove")
        self.result_text.grid(column=0, row=5, columnspan=2, pady=16, padx=8, sticky=tk.NSEW)

        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(5, weight=1)

        # Store league artifacts
        self.league_artifacts = {}

    def on_league_change(self, event=None):
        code = self.league_var.get().split(":")[0]
        csv_file = self.league_files.get(code)
        if not csv_file:
            self.home_cb['values'] = []
            self.away_cb['values'] = []
            self.df_raw = None
            return
        # Load and train if not already
        import os
        from logic import build_and_train
        if code not in self.league_artifacts:
            if not os.path.exists(csv_file):
                self.home_cb['values'] = []
                self.away_cb['values'] = []
                self.df_raw = None
                self.artifacts = None
                self.accuracy_var.set("Test accuracy: N/A")
                return
            artifacts, acc = build_and_train(csv_file)
            self.league_artifacts[code] = (artifacts, acc)
        else:
            artifacts, acc = self.league_artifacts[code]
        self.artifacts = artifacts
        self.model = artifacts['model']
        self.le_home = artifacts['le_home']
        self.le_away = artifacts['le_away']
        self.features = artifacts['features']
        self.df_raw = artifacts['df_full']
        # Update teams
        teams = sorted(pd.unique(pd.concat([self.df_raw['HomeTeam'], self.df_raw['AwayTeam']])))
        self.home_cb['values'] = teams
        self.away_cb['values'] = teams
        self.home_var.set(teams[0] if teams else "")
        self.away_var.set(teams[1] if len(teams) > 1 else "")
        self.accuracy_var.set(f"Test accuracy: {acc:.3f}")


    def predict(self):
        home = self.home_var.get()
        away = self.away_var.get()
        if not hasattr(self, 'df_raw') or self.df_raw is None or self.artifacts is None:
            messagebox.showwarning("Input error", "Please select a league with available data.")
            return
        if home == "" or away == "":
            messagebox.showwarning("Input error", "Pick both home and away teams.")
            return
        if home == away:
            messagebox.showwarning("Input error", "Home and Away must be different.")
            return

        h_stats = latest_team_stats_from_df(self.df_raw, home, n=5)
        a_stats = latest_team_stats_from_df(self.df_raw, away, n=5)
        if h_stats is None or a_stats is None:
            messagebox.showerror("No history", "One of the teams has no previous matches in dataset.")
            return

        feature_vector = []
        for feat in self.features:
            if feat == 'home_pts_last5_avg':
                feature_vector.append(h_stats['pts_avg'])
            elif feat == 'away_pts_last5_avg':
                feature_vector.append(a_stats['pts_avg'])
            elif feat == 'home_gd_last5_avg':
                feature_vector.append(h_stats['gd_avg'])
            elif feat == 'away_gd_last5_avg':
                feature_vector.append(a_stats['gd_avg'])
            elif feat == 'home_gf_last5_avg':
                feature_vector.append(h_stats['gf_avg'])
            elif feat == 'away_gf_last5_avg':
                feature_vector.append(a_stats['gf_avg'])
            elif feat in ['B365H','B365D','B365A']:
                if feat in self.df_raw.columns:
                    val = self.df_raw[feat].replace('', np.nan).astype(float).mean()
                    feature_vector.append(float(val) if not pd.isna(val) else 0.0)
                else:
                    feature_vector.append(0.0)
            elif feat == 'HomeEnc':
                try:
                    feature_vector.append(self.le_home.transform([home])[0])
                except Exception:
                    feature_vector.append(0)
            elif feat == 'AwayEnc':
                try:
                    feature_vector.append(self.le_away.transform([away])[0])
                except Exception:
                    feature_vector.append(0)
            else:
                feature_vector.append(0.0)

        X_in = np.array(feature_vector).reshape(1, -1)
        pred = self.model.predict(X_in)[0]
        pred_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
        pred_full = pred_map.get(pred, str(pred))

        if hasattr(self.model, "predict_proba"):
            prob_vals = self.model.predict_proba(X_in)[0]
            classes = self.model.classes_
            prob_map = {pred_map.get(c, c): round(float(prob_vals[i]), 3) for i, c in enumerate(classes)}
        else:
            prob_map = {}

        out = "\n=== Prediction Result ===\n"
        out += f"Predicted Outcome: {pred_full}\n"
        if prob_map:
            out += "\nProbabilities:\n"
            for k, v in prob_map.items():
                out += f"  {k}: {v*100:.1f}%\n"
        out += "\n--- Feature Values Used ---\n"
        for f, v in zip(self.features, feature_vector):
            if f.startswith('B365'):
                continue  # Skip Bet365 Odds features in log
            f_disp = f.replace('home_', 'Home ').replace('away_', 'Away ').replace('_last5_avg', ' (Last 5 Avg)').replace('gf', 'Goals For').replace('gd', 'Goal Diff').replace('pts', 'Points').replace('Enc', ' (Encoded)')
            out += f"{f_disp}: {v}\n"
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, out)

def main():
    root = tk.Tk()
    # No artifacts at start; user must select league
    app = PredictorApp(root, artifacts=None)
    root.title("Football Predictor")
    root.mainloop()

if __name__ == "__main__":
    main()
