# Database including the protocol for Static Fixed BT-LDA and Adaptive CC sLDA which change their training set when facing a session with a new condition

static_protocol = {
    # id : {
    #       patient_nr,
    #       last_session,
    #       selection_for_calibration (e.g. "6D_long_350")
    #       changing_condition: True if this patient has sessions with different conditions, False otherwise
    #       
    #       changing_starter_sessions (only if changing_condition is True): each session where a new condition starts
    #       }

    1: {
        "patient_nr": 1, 
        "last_session": 18,
        "selection": "6D_long_350",
        "changing_condition": False
    },
    2: {
        "patient_nr": 2, 
        "last_session": 14,
        "selection": "6D_long_350",
        "changing_condition": False
    },
    3: {
        "patient_nr": 3, 
        "last_session": 17,
        "selection": "6D_short_250",
        "changing_condition": True,

        "changing_starter_sessions":{
            3:"6D_short_250",
            11: "HP_short_250"
        }
    },
    4: {
        "patient_nr": 4, 
        "last_session": 20,
        "selection": "6D_long_350",
        "changing_condition": True,

        "changing_starter_sessions":{
            3:"6D_long_350",
            12: "HP_long_350"
        }
    },
    5: {
        "patient_nr": 5, 
        "last_session": 14,
        "selection": "6D_short_250",
        "changing_condition": True,
        
        "changing_starter_sessions":{
            3:"6D_short_250",
            8: "HP_long_350",
            11: "HP_short_250"
        }
    },
    6: {
        "patient_nr": 6, 
        "last_session": 16,
        "selection": "6D_short_250",
        "changing_condition": True,


        "changing_starter_sessions":{
            3:"6D_short_250",
            9: "HP_short_250"
        }
    },
    7: {
        "patient_nr": 7, 
        "last_session": 28,
        "selection": "6D_long_350",
        "changing_condition": False
    },
    8: {
        "patient_nr": 8, 
        "last_session": 19,
        "selection": "6D_short_250",
        "changing_condition": True,
        
        "changing_starter_sessions":{
            4:"6D_short_250",
            11: "HP_short_250"
        }
    },
    9: {
        "patient_nr": 9, 
        "last_session": 18,
        "selection": "6D_long_350",
        "changing_condition": True,
        
        "changing_starter_sessions":{
            3:"6D_long_350",
            11: "HP_long_350",
        }
    },
    10: {
        "patient_nr": 10, 
        "last_session": 16,
        "selection": "6D_long_350",
        "changing_condition": False
    }
}