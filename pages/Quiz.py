import streamlit as st
from services.quiz_service import QuizService
from components.styles import load_css

st.set_page_config(page_title="AI Quiz Generator", page_icon="📝", layout="wide")
load_css()

service = QuizService()

# ---- session state ----
_defaults = {
    "quiz": None,
    "current_question": 0,
    "score": 0,
    "answers": {},
    "quiz_submitted": False,
    "current_quiz_review": None,
    "review_error": None,
    "difficulty": "Medium",
    "question_type": "Multiple Choice",
    "num_questions": 5,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Latest quiz result (shared with the Reflection page)
_latest_defaults = {
    "quiz_completed": False,
    "quiz_score": None,
    "quiz_total": None,
    "quiz_correct": None,
    "quiz_wrong": None,
    "quiz_percentage": None,
    "quiz_status": None,
    "quiz_review": None,
    "quiz_overall_performance_summary": None,
    "quiz_agent_message": None,
}
for k, v in _latest_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _store_latest_quiz(score_result, review=None):
    st.session_state.quiz_completed = True
    st.session_state.quiz_score = score_result["score"]
    st.session_state.quiz_total = score_result["total"]
    st.session_state.quiz_correct = score_result["score"]
    st.session_state.quiz_wrong = score_result["total"] - score_result["score"]
    st.session_state.quiz_percentage = score_result["percentage"]
    st.session_state.quiz_status = score_result["status"]
    st.session_state.quiz_review = review
    st.session_state.quiz_overall_performance_summary = (
        review.get("overall_learning_report", {}).get("overall_performance")
        if review else None
    )
    st.session_state.quiz_agent_message = service.create_reflection_message(score_result, review)


def _do_submit(quiz):
    score_result = service.calculate_score(quiz, st.session_state.answers)
    st.session_state.score = score_result["score"]
    review_result = service.review_quiz(quiz, st.session_state.answers, provider="auto")
    if review_result["success"]:
        st.session_state.current_quiz_review = review_result["review"]
        st.session_state.review_error = None
        _store_latest_quiz(score_result, review_result["review"])
    else:
        st.session_state.current_quiz_review = None
        st.session_state.review_error = review_result["message"]
        _store_latest_quiz(score_result)
    st.session_state.quiz_submitted = True
    st.session_state.current_question = len(quiz)
    st.rerun()


st.title("AI Quiz Generator")
st.caption("Test yourself with a quiz generated from your uploaded notes.")

quiz = st.session_state.quiz

# ==================== SETUP ====================
if quiz is None:
    with st.container(border=True):
        st.markdown("### Quiz Settings")

        c1, c2, c3 = st.columns(3)
        with c1:
            difficulty = st.selectbox(
                "Difficulty", ["Easy", "Medium", "Hard"], key="difficulty"
            )
        with c2:
            question_type = st.selectbox(
                "Question Type", ["Multiple Choice", "True / False"], key="question_type"
            )
        with c3:
            num_questions = st.slider(
                "Number of Questions", 5, 20, step=5, key="num_questions"
            )

        if st.button("Generate Quiz", type="primary", use_container_width=True):
            with st.spinner("Generating your quiz..."):
                result = service.generate_quiz(
                    difficulty=difficulty,
                    question_type=question_type,
                    num_questions=num_questions,
                )
            if result["success"]:
                st.session_state.quiz = result["quiz"]
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.current_quiz_review = None
                st.session_state.review_error = None
                st.rerun()
            else:
                st.error(result["message"])
    st.stop()

total = len(quiz)
index = st.session_state.current_question

# ==================== COMPLETED ====================
if index >= total:
    percentage = (st.session_state.score / total) * 100 if total else 0
    status = "PASS" if percentage >= 50 else "FAIL"

    st.markdown(f"## {'🎉' if status == 'PASS' else '📚'} Quiz Complete")

    m1, m2, m3 = st.columns(3)
    m1.metric("Score", f"{st.session_state.score} / {total}")
    m2.metric("Percentage", f"{percentage:.0f}%")
    m3.metric("Status", status)

    if st.session_state.review_error:
        st.error(st.session_state.review_error)

    review = st.session_state.current_quiz_review
    if review:
        st.markdown("---")
        st.markdown("### Review")
        for item in review.get("questions", []):
            correct = item.get("status", "").strip().lower() == "correct"
            label = "✅ Correct" if correct else "❌ Incorrect"
            with st.expander(f"Question {item.get('question_number', '?')} — {label}"):
                st.markdown(f"**Your answer:** {item.get('your_answer', '')}")
                st.markdown(f"**Correct answer:** {item.get('correct_answer', '')}")
                st.markdown(f"**Explanation:** {item.get('explanation', '')}")

        report = review.get("overall_learning_report", {})
        if report:
            st.markdown("### Learning Report")
            for title, key in [
                ("Overall Performance", "overall_performance"),
                ("Strengths", "strengths"),
                ("Weak Areas", "weak_areas"),
                ("Topics to Revise", "topics_to_revise"),
                ("Study Recommendations", "study_recommendations"),
                ("Motivation", "motivational_feedback"),
                ("Readiness Level", "estimated_readiness_level"),
            ]:
                if key in report:
                    st.markdown(f"**{title}:** {report[key]}")

    st.markdown("---")
    if st.button("Generate New Quiz", type="primary", use_container_width=True):
        st.session_state.quiz = None
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.current_quiz_review = None
        st.session_state.review_error = None
        st.rerun()
    st.stop()

# ==================== QUESTION ====================
q = quiz[index]
st.progress((index + 1) / total)

option_keys = list(q["options"].keys())
option_labels = [f"{k}. {q['options'][k]}" for k in option_keys]
label_to_key = dict(zip(option_labels, option_keys))

saved = st.session_state.answers.get(index)
saved_idx = option_keys.index(saved) if saved in option_keys else None

with st.container(border=True):
    st.markdown(f"**Question {index + 1} of {total}**")
    st.markdown(f"### {q['question']}")
    selected = st.radio(
        "Choose your answer",
        option_labels,
        index=saved_idx,
        key=f"radio_{index}",
        label_visibility="collapsed",
    )
    if selected:
        st.session_state.answers[index] = label_to_key[selected]

answered = len(st.session_state.answers)
st.caption(f"**{answered} / {total} answered**")

# ---- navigation ----
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("← Previous", disabled=(index == 0), type="secondary", use_container_width=True):
        st.session_state.current_question -= 1
        st.rerun()
with nav2:
    if index < total - 1:
        if st.button("Next →", type="primary", use_container_width=True):
            st.session_state.current_question += 1
            st.rerun()

# ---- submit (full width, only on last question) ----
if index == total - 1:
    st.markdown("---")
    all_answered = answered == total
    if not all_answered:
        st.warning(f"You've answered {answered} of {total} questions. Answer them all to submit.")
    if st.button("Submit Quiz", type="primary", disabled=not all_answered, use_container_width=True):
        with st.spinner("Reviewing your answers..."):
            _do_submit(quiz)
