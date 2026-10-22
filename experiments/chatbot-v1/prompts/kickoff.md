Kickoff (chatbot-v1). 8 agents, 4 hours, ONE conversational chatbot: natural, friendly small talk like a curious friend, NOT a therapist (no "how does that make you feel", no turning the user's words into questions).
Phase 1 (first 2 h): two teams work separately on different directions.
- Team A (ds-a-0000, ds-a-0001, glm-a-0000, glm-a-0001): persona + small-talk engine with memory of the user. Thread: [team-a].
- Team B (ds-b-0000, ds-b-0001, glm-b-0000, glm-b-0001): intent-driven dialogue with topic threads and follow-up questions. Thread: [team-b].
Each team posts only in its own thread and ignores the other team's. The world keeps the teams' files apart until the merge.
Phase 2 (last 2 h): the teams merge, everything becomes readable, promote_best opens, and all 8 build best/bot.py together. A final round ends the run with GREEN/RED verdicts.
Do not reply to this kickoff.
