# Altea Resort Booking System - Detailed Code Reference

This document explains the current codebase in detail: file-by-file purpose, module responsibilities, and function-level behavior for backend and frontend.

---

## 1. System Architecture (Current)

The system is a Flask monolith organized by blueprints:

- **Public/User flows:** home pages, guest booking, booking confirmation/status, cancellation.
- **Auth flows:** user/admin registration and login.
- **Admin flows:** dashboard, booking operations, notifications, analytics (realtime + BI reports + executive view), feedback moderation.
- **Integrations:** Supabase (DB/storage/auth), Redis (session/drafts/cache), Google Calendar, Google Drive, Gemini/Langbly for sentiment and text processing, SMTP email.

Primary runtime entry point:

- `main.py` creates and configures Flask app, session backend, mail backend, and blueprint registration.

---

## 2. Root Files

### `main.py`
- **Role:** application bootstrap and configuration.
- **Important code:**
  - Registers all blueprints under dedicated prefixes (`/guest_booking`, `/booking_confirm`, `/admin`, etc.).
  - Configures `SESSION_TYPE='redis'` via shared Redis client.
  - Configures Gmail SMTP for `Flask-Mail`.
  - Defines Jinja filters:
    - `format_date(value)`
    - `format_date_month(value)`
  - Redirects `/` to `home.home_index`.

### `README.md`
- **Role:** setup and high-level project documentation.

### `analytics-ml-implementation.md`
- **Role:** implementation strategy for realtime analytics, BI, executive support, predictive ML feasibility, Plotly integration/data mapping.

### `summary.md`
- **Role:** running change log of implemented features/fixes and chart explanations.

### `todo.md`
- **Role:** project notes/todo list.

### `requirements.txt`
- **Role:** Python dependency manifest.

### `altea-booking-system-9bb45fa21aeb.json`, `client_secret_*.json`, `token*.json`, `oauth*_token.json`
- **Role:** credentials/tokens used by integrations.

### `StyleGuide1.PNG`, `StyleGuide2.PNG`
- **Role:** visual style references.

### `generate_token.py`
- **Role:** OAuth token generation helper consumed by service client initialization.

---

## 3. Backend Modules (`app/`)

## 3.1 Core Integration Layer

### `app/api_functions/api_services.py`
- **Role:** centralized service container (`ServiceClients`) for external systems.
- **Exports:** `services = ServiceClients()`

#### Class `ServiceClients`
- `__init__(self)`
  - Initializes Supabase client.
  - Loads Google service account path, calendar id, Gemini and Langbly keys.
  - Prepares lazy service handles.
- `_get_oauth_creds(self)`
  - Ensures valid OAuth credentials for Google Drive (refresh/regenerate as needed).
- `drive` (property)
  - Lazy-creates Google Drive API client.
- `calendar` (property)
  - Lazy-creates Google Calendar API client using service account.
- `gemini` (property)
  - Lazy-creates Gemini client.
- `gemini_generate_content(self, contents, model, **kwargs)`
  - Wrapper to Gemini text generation.
- `langbly_translate(self, text, source='auto', target='en')`
  - Calls Langbly translation API.
- `redis` (property)
  - Returns Redis client from `REDIS_URL`.

---

## 3.2 Booking Draft and Utility Modules

### `app/booking_drafts.py`
- **Role:** save/restore booking wizard progress in Redis.

Functions:
- `save_draft_to_redis(booking_id, session_data)`
  - Serializes selected booking/session fields and stores with 24-hour TTL.
- `get_draft_from_redis(booking_id)`
  - Restores draft JSON for a booking id.
- `delete_draft_from_redis(booking_id)`
  - Removes draft after successful booking confirmation.

### `app/uuid_generator.py`
- **Role:** deterministic ID helpers.

Functions:
- `user_id_generator(name)`
  - UUIDv5 from name.
- `payment_id_generator(booking_id)`
  - UUIDv5 from booking id.

### `app/extensions.py`
- **Role:** shared Flask extension object.
- Defines `mail = Mail()` for app initialization in `main.py`.

---

## 3.3 File Upload Integrations

### `app/api_functions/google_drive_functions.py`
- **Role:** upload proofs to Google Drive and return link/id.

Function:
- `add_file_proof(full_name, booking_id, file_session_key)`
  - Reads proof from Flask session (or Redis draft fallback).
  - Chooses target folder by proof type (`id_proof`, `downpayment_proof`).
  - Uploads file via Drive API and sets public read permission.
  - Returns `(webViewLink, file_id)`.

### `app/supabase_bucket.py`
- **Role:** upload proofs to Supabase Storage.

Function:
- `supabase_add_file_proof(full_name, booking_id, file_session_key)`
  - Reads proof from session/draft.
  - Converts session hex data to bytes.
  - Uploads to `booking-proofs` bucket and returns `(public_url, storage_path)`.

### `app/photo_preview.py`
- **Role:** temporary session image preview endpoint.

Function:
- `preview(filename)`
  - Reads hex blob from session and streams it as JPEG.

---

## 3.4 Booking + Status + Cancellation Data Helpers

### `app/db_booking_functions.py`
- **Role:** booking token validation and joined booking context fetch.

Function:
- `check_booking_with_token(booking_id, token)`
  - Loads booking by id.
  - Validates access token.
  - Loads latest downpayment payment row.
  - Loads linked guest/user row.
  - Returns tuple `(booking, payment_data, guest_data, user_data)` or HTTP error response.

### `app/get_bookings_and_info.py`
- **Role:** enrichment layer for booking/cancellation displays and admin views.

Functions:
- `_safe_name(first_name, last_name, fallback='Guest')`
  - Normalizes name assembly.
- `_pick_preferred_payment(current_payment, candidate_payment)`
  - Prefers downpayment and/or newer payment rows.
- `get_bookings_with_guest_names()`
  - Bulk-fetches bookings, users, guests, payments.
  - Enriches each booking with guest identity and payment metadata.
- `selected_booking_details(booking_id)`
  - Returns one enriched booking by id.
- `get_cancellations_with_guest_names()`
  - Enriches cancellation rows using booking/guest context.
- `selected_cancellation_details(cancellation_id)`
  - Returns one enriched cancellation row by id.

---

## 3.5 Email Module

### `app/emails.py`
- **Role:** centralized email sending + admin broadcast.

Functions:
- `send_email(recipient, subject, body='', status_link=None, sender=None, html_content=None)`
  - Sends plain/html email through Flask-Mail.
- `get_admin_recipients(include_system_sender=True)`
  - Loads active admin emails from Supabase (+ optional system sender).
- `notify_admins(subject, body, html_content=None, status_link=None)`
  - Broadcast helper; sends same message to all admin recipients.

---

## 3.6 Home/Public Blueprint

### `app/home.py`
- **Blueprint:** `home_bp`
- **Role:** public pages, feedback, user profile, admin-dashboard bridge.

Functions/routes:
- `home_index(role=None, username=None)` -> `/home/` and `/home/<role>`
  - Initializes default session role (`guest`) and renders landing page.
- `contacts()` -> `/home/contacts`
- `about()` -> `/home/about`
- `booking()` -> `/home/booking`
  - Redirects to guest booking flow.
- `customer_feedback()` -> `/home/feedback`
  - Handles feedback submission for guest/user; admin/employee gets redirected to admin feedback page.
  - Uses sentiment analyzer before writing comments to DB.
- `admin_dashboard(admin_id, role)` -> `/home/admin/...`
  - Guards admin session and redirects to admin dashboard.
- `user_profile()` -> `/home/profile`
  - Shows/updates user profile and splits bookings into active/history.

---

## 3.7 Guest Booking Blueprint

### `app/guest_booking.py`
- **Blueprint:** `guest_booking_bp`
- **Role:** multi-step booking wizard and booked-date availability endpoint.

Functions/routes:
- `_safe_float(value, default=0.0)`
- `_compute_total_amount(base_rate, check_in_date, check_out_date)`
  - Computes nights and total amount from slot base rate.
- `clear_session_keys(role, booking_id)`
  - Clears step/session fields per role.
- `guest_booking_index()` -> `/guest_booking/`
  - Generates booking id and redirects to booking landing.
- `guest_booking_route(booking_id)` -> `/guest_booking/<booking_id>`
  - Shows landing page for that booking id.
- `guest_booking_start(booking_id)` -> `/guest_booking/<booking_id>/start-booking`
  - Main multi-step flow:
    - **Step 1:** date selection, slot details, total amount calculation.
    - **Step 2:** personal info + ID proof (with terms gating).
    - **Step 3:** payment metadata + downpayment proof.
    - **Step 4:** redirect to confirmation module.
  - Persists intermediate progress to Redis drafts.
- `reset(booking_id)` -> `/guest_booking/<booking_id>/reset`
- `get_booked_dates()` -> `/guest_booking/booked-dates`
  - Returns non-cancelled ranges for calendar blocking.
- `go_back(booking_id)` -> `/guest_booking/<booking_id>/back`
  - Moves booking wizard one step back safely.

**Important logic recently stabilized:**
- Total amount now derives from computed nights and slot base rate.
- Session + draft synchronization avoids data loss when navigating steps.

---

## 3.8 Booking Confirmation / Status Blueprint

### `app/guest_booking_confirm.py`
- **Blueprint:** `guest_booking_confirm_bp`
- **Role:** finalize booking inserts, status pages, reschedule, proof reupload.

Functions/routes:
- `_safe_float(value, default=0.0)`
- `_resolve_total_amount(session_data)`
  - Defensive fallback total computation (session total -> slot base_rate * days).
- `booking_confirmation_view(booking_id)` -> `/booking_confirm/<booking_id>/booking_confirmation_view`
  - Renders final confirmation review with proof previews.
- `booking_confirmation(booking_id)` -> `/booking_confirm/<booking_id>/booking_confirmation`
  - Final commit path:
    - Uploads proof files to Google Drive and Supabase bucket.
    - Creates/links calendar event.
    - Inserts `payments`, `guest_users`/updates `users`, and `bookings`.
    - Stores secure access token for status page.
    - Sends guest email + admin notifications.
- `confirmed_booking(booking_id)` -> `/booking_confirm/<booking_id>/submitted-booking`
- `booking_status(booking_id)` -> `/booking_confirm/status/<booking_id>/`
  - Token-guarded booking status view with payment and booking details.
- `_upload_status_file(file_obj, booking_id, suffix)`
  - Shared uploader for reupload flows.
- `booking_reschedule(booking_id)` -> `/booking_confirm/status/<booking_id>/reschedule`
  - Customer reschedule request route; sets booking back to pending and notifies admins.
- `reupload_id_proof(booking_id)` -> `/booking_confirm/status/<booking_id>/reupload-id`
  - Replaces ID proof in user/guest table and updates booking remarks/status.
- `reupload_payment_proof(booking_id)` -> `/booking_confirm/status/<booking_id>/reupload-payment`
  - Updates/inserts payment proof and marks status pending.

**Important logic:**
- Booking inserts now consistently set `total_amount`.
- Status/reupload actions preserve token-based access controls.

---

## 3.9 Booking Cancellation Blueprint

### `app/booking_cancellation.py`
- **Blueprint:** `guest_booking_cancellation_bp`
- **Role:** cancellation request workflow with token security and cancellation fee proof.

Functions/routes:
- `_upload_cancellation_proof(file_obj, cancellation_id)`
  - Uploads cancellation proof to Supabase storage bucket.
- `cancellation_index(booking_id)` -> `/booking_cancellation/<booking_id>`
  - Validates token, resolves/creates cancellation id, redirects to form.
- `cancel_booking(booking_id, cancellation_id)` -> `/booking_cancellation/cancel_request/...`
  - Renders cancellation form.
- `cancellation_request(booking_id, cancellation_id)` -> `/booking_cancellation/cancel_request/.../confirm`
  - Inserts/updates cancellation request.
  - Inserts cancellation payment row.
  - Updates booking remarks.
  - Sends guest + admin emails.
- `cancellation_submitted(cancellation_id, booking_id)`
  - Submission acknowledgement page.
- `cancellation_status(booking_id, cancellation_id)`
  - Token-guarded cancellation status page.

---

## 3.10 Auth Blueprints

### `app/auth/check_user_exists.py`
- `check_user_exists(email)`
  - Checks both `users` and `admins` tables for duplicate emails.

### `app/auth/db_add_user.py`
- `add_client_db(...)`
  - Inserts app user row with hashed password.
- `add_admin_db(...)`
  - Inserts admin row with hashed password and role.

### `app/auth/register.py`
- **Blueprint:** `register_bp`

Routes:
- `register_page()` -> `/register/client`
- `register_client()` -> `/register/client/submit`
  - Supabase auth sign-up + app table insertion.
  - Can convert/link guest booking to newly registered user.
- `register_admin_page()` -> `/register/admin/...`
- `register_admin()` -> `/register/admin/submit/...`
  - Admin account creation flow with role assignment.

### `app/auth/login.py`
- **Blueprint:** `login_bp`

Routes:
- `login_page()` -> `/login/`
- `admin_login_page()` -> `/login/admin`
- `login_submit()` -> `/login/submit`
  - Authenticates via Supabase auth.
  - Resolves role by checking `admins` table.
  - Redirects admin/users to correct landing.
- `logout()` -> `/login/logout`
  - Clears session and resets role to `guest`.

---

## 3.11 Admin Notification Blueprint

### `app/admin_function/admin_notification.py`
- **Blueprint:** `admin_notif_bp`
- **Role:** admin review/approval actions for booking and cancellation queues.

Functions/routes:
- `require_admin_session()` (`before_request`)
  - Enforces logged-in admin context and route/session consistency.
- `notif_home(admin_id, role)`
  - Combined list view for booking + cancellation notifications.
- `detailed_view(admin_id, role, booking_id)`
  - Booking detailed review page.
- `detailed_view_submit(admin_id, role, booking_id)`
  - Persists booking status and payment review decisions.
  - Upserts Google Calendar event based on new booking status.
  - Sends guest/admin notification emails.
- `cancellation_detailed_view(admin_id, role, cancellation_id)`
  - Cancellation review page.
- `cancellation_detailed_view_submit(admin_id, role, cancellation_id)`
  - Persists cancellation decision.
  - Cancels related booking if approved.
  - Removes Google Calendar event for approved cancellations.
  - Sends guest/admin notifications.

---

## 3.12 Admin Main Blueprint

### `app/admin_function/admin.py`
- **Blueprint:** `admin_bp`
- **Role:** admin dashboard, booking management, analytics/BI/executive views, profile, feedback, exports.

Functions:
- `require_admin_session()` (`before_request`)
  - Global admin access guard.
- `_safe_date(value)`, `_parse_form_date(value)`
  - Date parsing helpers.
- `_has_blocking_booking(check_in_date, check_out_date)`
  - Detects overlap with pending/confirmed bookings.
- `_decision_level(value, green_threshold, yellow_threshold, higher_is_better=True)`
  - Traffic-light scoring helper.
- `_build_executive_payload()`
  - Builds risk flags + prioritized recommendations from realtime analytics.
- `_snapshot_to_csv_rows(snapshot)`
  - Flattens BI snapshot to CSV rows.
- `_fetch_raw_dataset_rows(dataset)`
  - Table whitelist + row fetch for raw export.
- `_rows_to_csv_response(rows, filename_prefix)`
  - Generic CSV response serializer.

Routes:
- `admin_page(admin_id, role)` -> dashboard
- `admin_availability(year)` -> calendar occupancy source data
- `admin_event_details(event_id)` -> Google event details
- `booking_notifications(admin_id, role)` -> redirect to notification module
- `booking_page(admin_id, role)` -> booking lists (active/history)
- `booking_create(admin_id, role)` -> admin-created bookings
  - Uses same blocked-date model as guest flow.
  - Server-side overlap validation prevents conflicts.
  - Auto-computes `total_amount` from active slot base rate.
- `analytics_page(admin_id, role)` -> unified analytics tab with mode selector
  - `view=realtime|reports|executive`
  - Realtime mode: live metrics/charts + monthly sentiment summary.
  - Reports mode: BI snapshot and report refresh/export actions.
  - Executive mode: KPI flags and recommendation cards.
- `analytics_realtime_data(admin_id, role)` -> JSON for chart polling
- `reports_page(admin_id, role)` -> legacy route redirect to analytics reports mode
- `download_report(admin_id, role, report_type)` -> BI CSV export
- `export_raw_dataset_csv(admin_id, role, dataset)` -> raw DB CSV export
- `executive_page(admin_id, role)` -> legacy route redirect to analytics executive mode
- `feedback_page(admin_id, role)` -> paginated feedback with author enrichment
- `admin_profile(admin_id, role)` -> admin profile read/update

**Important code additions in this phase:**
- Reports and executive were consolidated into analytics mode toggles.
- Raw CSV dataset export endpoint for BI transparency.
- Booking overlap/date validation and admin booking UX parity support.

---

## 3.13 Analytics and BI Modules

### `app/api_functions/bi_reporting.py`
- **Role:** analytics computation engine for realtime dashboards and BI snapshots.

Core constants:
- `COMPLAINT_THEME_KEYWORDS` maps theme -> keyword tuple for recurring complaint extraction.

Utility functions:
- `_safe_float(value, default=0.0)` (handles currency-formatted strings too)
- `_parse_datetime(value)`
- `_as_utc_aware(dt_value)` (naive/aware normalization)
- `_parse_date(value)`
- `_date_label(value)`, `_month_label(value)`
- `_comment_theme(text)` (keyword-based theme tagging)
- `_pick_booking_amount(booking)` (total -> payment amount -> down_payment fallback)
- `_pick_booking_revenue_day(booking, fallback_datetime=None)`
- `_estimate_amount_from_slot_rate(booking, slot_rates)` (final fallback estimation)
- `_fetch_payments()`
- `_fetch_booking_slot_rates()`
- `_fetch_comments()`
- `_build_daily_series(days)`
- `_build_month_axis(months)`

Primary functions:
- `build_realtime_payload(days=30, sentiment_months=6)`
  - Aggregates cards + chart series:
    - bookings trend (created/confirmed/cancelled),
    - daily confirmed revenue,
    - status/channel mix,
    - payment aging and revenue-at-risk,
    - occupancy outlook,
    - sentiment trend,
    - complaint themes.
  - Contains timezone-safe payment aging math.
  - Contains multi-level revenue fallback logic so confirmed revenue remains visible when raw amount fields are incomplete.
- `build_bi_snapshot()`
  - Produces structured BI snapshot (`booking_funnel`, channel mix, payment performance, feedback trends, KPI summary).
- `persist_bi_snapshot(snapshot)`
  - Writes daily/monthly snapshots into `bi_daily_kpis` and `bi_monthly_summary`.
- `run_bi_aggregation_job()`
  - Builds and persists snapshot in one call.

### `app/api_functions/bi_scheduler.py`
- **Role:** scheduler entry wrappers.

Functions:
- `run_daily_bi_job()`
- `run_weekly_bi_job()`
  - Both execute `run_bi_aggregation_job()`.

---

## 3.14 Calendar + Sentiment Support Modules

### `app/api_functions/google_calendar_functions.py`
- `_normalize_date(value)`
- `_build_event_payload(booking_id, booking_summary, check_in_date, check_out_date, status)`
- `add_booking_gcalendar(...)`
- `upsert_booking_gcalendar(...)`
  - Creates/updates/deletes booking events based on status and event id.
- `delete_booking_gcalendar(event_id)`
- `check_availability(year)`
  - Returns non-cancelled booking date ranges for admin calendar.
- `check_event_details(event_id)`

### `app/api_functions/sentiment_analysis.py`
- **Role:** per-comment sentiment + monthly sentiment summary generation with cache.

Functions:
- `_current_month_key(now=None)`
- `_month_range(month_key)`
- `_safe_redis()`
- `_summary_cache_key(month_key)`
- `_parse_response_text(response)`
- `_fetch_monthly_comments(month_key)`
- `_build_monthly_summary_payload(month_key, comments)`
- `get_monthly_sentiment_summary(month_key=None, force_refresh=False)`
  - Loads from Redis cache when possible, otherwise generates summary via Gemini.
- `translate(text)` (Langbly fallback for non-English)
- `ask_gemini(prompt, model='gemini-2.5-flash')`
- `_roberta_sentiment_analysis(text)` (transformer inference)
- `_analyze_comment_sentiment(comment_text, rating=None)`
  - Returns sentiment score/label/model metadata used during feedback insert.

---

## 4. Templates (`templates/`)

## 4.1 Shared/Public Templates

- `base.html`: global layout shell for public pages.
- `index.html`: homepage.
- `about.html`, `contacts.html`: informational pages.
- `feedback.html`: customer feedback submission form.
- `404.html`, `unauthorized.html`: error/access pages.
- `user_profile.html`: user profile + booking history/active bookings.

## 4.2 Booking Flow Templates

- `booking_landing_page.html`: start gateway for booking flow.
- `booking_page.html`: multi-step booking wizard UI.
- `booking_confirmation.html`: review-and-confirm summary with previews.
- `booking_done.html`: success page after submission.
- `booking_status.html`: token-gated status tracker.
- `booking_reschedule.html`: customer reschedule request UI.

## 4.3 Cancellation Templates

- `booking_cancellation.html`: cancellation request form.
- `booking_cancellation_done.html`: cancellation submitted acknowledgement.
- `booking_cancellation_status.html`: token-gated cancellation status tracker.

## 4.4 Auth Templates (`templates/auth/`)

- `base_auth.html`: auth page wrapper.
- `login.html`: user/admin login mode UI.
- `client_register.html`: client registration.
- `admin_register.html`: admin registration (protected path).

## 4.5 Email Templates (`templates/email/`)

- `email_base.html`: shared HTML email layout.
- `booking_details.html`: booking lifecycle email body.
- `cancellation_details.html`: cancellation lifecycle email body.

## 4.6 Admin Templates (`templates/admin_function/`)

- `admin_base.html`: admin layout + sidebar.
- `admin_dashboard.html`: admin home metrics + mini realtime charts.
- `admin_bookings.html`: active/history booking list view.
- `admin_booking_create.html`: admin booking create form with guest-style date selection UX.
- `admin_notification.html`: booking/cancellation queue list.
- `admin_notif_detailed_view.html`: booking decision page.
- `admin_cancellation_detailed_view.html`: cancellation decision page.
- `admin_feedback.html`: comment management page with pagination.
- `admin_profile.html`: admin profile editor.
- `admin_analytics.html`: **primary analytics screen** with mode buttons:
  - realtime,
  - BI reports,
  - executive decision support.
- `admin_reports.html`, `admin_executive.html`: legacy/supplemental analytics pages retained in repo.

---

## 5. Frontend Scripts (`static/js/`)

### `calendar_booking.js`
- Guest/admin booking popup calendar engine:
  - fetches blocked date ranges,
  - prevents selecting ranges crossing pending/confirmed bookings,
  - controls check-in/check-out interaction states.

Functions:
- `toDate`, `toStr`, `getStatus`, `hasBlockedDateInRange`, `openCalendar`, `closeCalendar`, `changeMonth`, `renderCalendar`, `selectDate`, `updateDisplay`.

### `admin_booking_create.js`
- Hooks admin create form to calendar-based date fields and auto total calculation.

Functions:
- IIFE initializer,
- inner `toDate`,
- `recomputeTotalAmount`,
- wraps global `updateDisplay` to recalc amount after date changes.

### `calendar.js`
- Admin yearly occupancy calendar renderer for dashboard.

Functions:
- `toDateObj`,
- `getDateStatus`,
- `buildMonth`,
- `loadYear`,
- plus prev/next year handlers.

### `terms_dropdown.js`
- Terms/house-rules acknowledgment UI state.

Functions:
- `ensureDropdownState`,
- `toggleDropdown`,
- `markAsRead`,
- `updateSubmitState`.

### `admin_notification_detail.js`
- Auto-fills suggested remarks templates based on selected booking status on detail page.

### `admin_dashboard_realtime.js`
- Realtime mini-dashboard cards and charts on admin home.

Functions:
- `phpFormat`,
- `renderDashboardRealtime`,
- init IIFE with 60-second polling.

### `admin_analytics_plotly.js`
- Main realtime analytics chart renderer.

Functions:
- `formatPhp`,
- `renderAnalyticsCharts`,
- `refreshRealtimeAnalytics`,
- init IIFE with endpoint polling.

Charts rendered:
- bookings trend,
- status mix,
- channel mix,
- payment aging,
- daily confirmed revenue,
- occupancy,
- sentiment trend,
- complaint themes.

### `admin_reports_plotly.js`
- BI snapshot chart renderer (funnel, mix, aging, sentiment, complaint themes).

### `admin_executive_plotly.js`
- Executive charts (occupancy outlook + combined revenue/bookings signal).

### `main.js`
- Present in repo; currently minimal/empty.

---

## 6. Styles and Static Assets

### `static/css/styles.css`
- Main site/admin styles including dashboard cards, analytics layout, mode buttons, chart containers, decision-level badges.

### `static/css/booking.css`
- Booking flow and date-picker/calendar styling.

### `static/css/auth_styles.css`
- Auth page styling.

### `static/css/email_styles.css`
- Email template styling.

### `static/assets/BookingsColumnNames.csv`
- Reference columns for booking dataset exports/analysis.

### `static/assets/PaymentsColumnNames.csv`
- Reference columns for payment dataset exports/analysis.

---

## 7. Key Data Flow Summary

## 7.1 Booking creation (guest)
1. Booking wizard captures dates, guest details, proofs, payment refs.
2. Draft state persists in Redis between steps.
3. Confirmation uploads files (Drive + Supabase), creates calendar event, inserts `bookings`/`payments`/guest data.
4. Access token enables secure status page.
5. Guest + admin email notifications are sent.

## 7.2 Admin booking-for-customer
1. Admin uses booking form with blocked-date calendar logic matching guest UX.
2. Server validates date range and overlap.
3. Booking is inserted with computed `total_amount`.
4. Optional payment row can be inserted.
5. Notification emails are sent.

## 7.3 Analytics/BI/Executive
1. `bi_reporting.py` aggregates bookings/payments/comments.
2. Realtime cards + Plotly series are served via JSON endpoint.
3. BI snapshot can be refreshed and exported to CSV.
4. Raw dataset export endpoint exposes table-level CSV for analytics validation.
5. Executive mode computes decision flags and recommendations from KPI thresholds.

---

## 8. Notes on Current Implementation Quality and Behavior

- **Timezone-safe analytics math** was added to prevent naive/aware datetime subtraction errors.
- **Revenue computation** uses layered fallbacks to avoid empty charts when amount fields are partially missing.
- **Reports + Executive navigation** was consolidated into analytics mode buttons to reduce admin sidebar confusion.
- **Raw data CSV export** is intentionally table-whitelisted for safer analytics access.

---

## 9. Complete Python Module Index

- `app/admin_function/admin.py`
- `app/admin_function/admin_notification.py`
- `app/api_functions/api_services.py`
- `app/api_functions/bi_reporting.py`
- `app/api_functions/bi_scheduler.py`
- `app/api_functions/google_calendar_functions.py`
- `app/api_functions/google_drive_functions.py`
- `app/api_functions/sentiment_analysis.py`
- `app/auth/check_user_exists.py`
- `app/auth/db_add_user.py`
- `app/auth/login.py`
- `app/auth/register.py`
- `app/booking_cancellation.py`
- `app/booking_drafts.py`
- `app/db_booking_functions.py`
- `app/emails.py`
- `app/extensions.py`
- `app/get_bookings_and_info.py`
- `app/guest_booking.py`
- `app/guest_booking_confirm.py`
- `app/home.py`
- `app/photo_preview.py`
- `app/supabase_bucket.py`
- `app/uuid_generator.py`

---

## 10. Complete Template Index

- `templates/404.html`
- `templates/about.html`
- `templates/base.html`
- `templates/booking_cancellation.html`
- `templates/booking_cancellation_done.html`
- `templates/booking_cancellation_status.html`
- `templates/booking_confirmation.html`
- `templates/booking_done.html`
- `templates/booking_landing_page.html`
- `templates/booking_page.html`
- `templates/booking_reschedule.html`
- `templates/booking_status.html`
- `templates/contacts.html`
- `templates/feedback.html`
- `templates/index.html`
- `templates/unauthorized.html`
- `templates/user_profile.html`
- `templates/auth/admin_register.html`
- `templates/auth/base_auth.html`
- `templates/auth/client_register.html`
- `templates/auth/login.html`
- `templates/email/booking_details.html`
- `templates/email/cancellation_details.html`
- `templates/email/email_base.html`
- `templates/admin_function/admin_analytics.html`
- `templates/admin_function/admin_base.html`
- `templates/admin_function/admin_booking_create.html`
- `templates/admin_function/admin_bookings.html`
- `templates/admin_function/admin_cancellation_detailed_view.html`
- `templates/admin_function/admin_dashboard.html`
- `templates/admin_function/admin_executive.html`
- `templates/admin_function/admin_feedback.html`
- `templates/admin_function/admin_notif_detailed_view.html`
- `templates/admin_function/admin_notification.html`
- `templates/admin_function/admin_profile.html`
- `templates/admin_function/admin_reports.html`

---

## 11. Complete Frontend Script Index

- `static/js/admin_analytics_plotly.js`
- `static/js/admin_booking_create.js`
- `static/js/admin_dashboard_realtime.js`
- `static/js/admin_executive_plotly.js`
- `static/js/admin_notification_detail.js`
- `static/js/admin_reports_plotly.js`
- `static/js/calendar.js`
- `static/js/calendar_booking.js`
- `static/js/main.js`
- `static/js/terms_dropdown.js`

