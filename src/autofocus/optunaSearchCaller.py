import json
import optuna

sampler = optuna.samplers.RandomSampler()
study = optuna.create_study(sampler=sampler, direction="minimize")

with open("search_bounds.json", "r") as search_bounds:
    with open("search_grid_points.csv", "w") as gridPoints:
        data = json.load(search_bounds)

        ntrials = data["ntrials"]
        for _ in range(ntrials):
            trial = study.ask()
            params = []
            for key in data.keys():
                if key == "ntrials":
                    continue
                param = trial.suggest_float(key, data[key]["low"], data[key]["high"])
                params.append(param)
            for param in params:
                gridPoints.write(str(param)+",")
            gridPoints.write("\n")
