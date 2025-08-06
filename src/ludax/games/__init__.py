# Package a subset of default game implementations

connect_six = """
(game "Connect-Six"
    (players 2)
    (equipment 
        (board (square 19))
    ) 
    
    (rules 
        (play
            (once-through (P1)
                (place (destination empty))
            )
            (repeat (P2 P2 P1 P1)
                (place (destination empty))
            )
        )
        (end 
            (if (line 6) (mover win))
        )
    )
)
"""

connect_four = """
(game "Connect-Four"
    (players 2)
    (equipment 
        (board (rectangle 6 7))
    ) 
    
    (rules 
        (play
            (repeat (P1 P2)
                (place (destination (and empty (or (edge bottom) (adjacent occupied direction:up)))))
            )
        )
        
        (end 
            (if (line 4) (mover win))
            (if (full_board) (draw))    
        )
    )
)
"""

hex = """
(game "Hex"
    (players 2)
    (equipment 
        (board (hex_rectangle 11 11))
    ) 
    
    (rules 
        (play
            (repeat (P1 P2)
                (place (destination empty))
            )
        )
        
        (end 
            (if (and (>= (connected ((edge top) (edge bottom))) 2) (mover_is P1)) (mover win))
            (if (and (>= (connected ((edge left) (edge right))) 2) (mover_is P2)) (mover win))
            (if (full_board) (draw))    
        )
    )
)
"""

tic_tac_toe = """
(game "Tic-Tac-Toe" 
    (players 2)
    (equipment 
        (board (square 3))
    ) 
    
    (rules 
        (play
            (repeat (P1 P2)
                (place (destination empty))
            )
        )
        
        (end 
            (if (line 3) (mover win))
            (if (full_board) (draw))    
        )
    )
)
"""

reversi = """
(game "Reversi" 
    (players 2)
    (equipment 
        (board (square 8))
    ) 
    
    (rules
        (start
            (place P1 (27 36))
            (place P2 (28 35))
        )
        (play
            (repeat (P2 P1)
                (place 
                    (destination empty)
                    (result 
                        (exists
                            (custodial any)
                        )
                    )
                    (effects 
                        (flip (custodial any))
                        (set_score mover (count (occupied mover)))
                        (set_score opponent (count (occupied opponent)))
                    )
                )
                (force_pass)
            )
        )
        
        (end
            (if (passed both) (by_score))    
        )
    )
)
"""
