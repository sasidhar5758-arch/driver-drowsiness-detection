def fatigue_stage(prediction):

    if prediction in ["open", "no_yawn"]:
        return "Alert"

    elif prediction == "yawn":
        return "Mild Fatigue"

    elif prediction == "closed":
        return "Severe Fatigue"