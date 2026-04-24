As a Senior Full-stack developer for Aelta Booking System, here’s the information you need in order to do the project:

# Database schemas:

create table public.activity_logs (
log_id uuid not null default extensions.uuid_generate_v4 (),
actor_type character varying(20) not null,
actor_id uuid not null,
action character varying(100) not null,
entity_type character varying(50) null,
entity_id uuid null,
description text null,
ip_address inet null,
user_agent text null,
created_at timestamp with time zone null default now(),
constraint activity_logs_pkey primary key (log_id)
) TABLESPACE pg_default;

create index IF not exists idx_activity_logs_actor on public.activity_logs using btree (actor_type, actor_id) TABLESPACE pg_default;

create index IF not exists idx_activity_logs_created_at on public.activity_logs using btree (created_at) TABLESPACE pg_default;

create table public.admins (
admin_id uuid not null default extensions.uuid_generate_v4 (),
first_name character varying(100) not null,
admin_role character varying(50) not null,
username character varying(50) not null,
password_hash character varying(255) not null,
email character varying(255) null,
created_at timestamp with time zone null default now(),
updated_at timestamp with time zone null default now(),
is_active boolean null default true,
last_name character varying not null,
phone_number character varying null,
constraint admins_pkey primary key (admin_id),
constraint admins_username_key unique (username)
) TABLESPACE pg_default;

create trigger update_admins_updated_at BEFORE
update on admins for EACH row
execute FUNCTION update_updated_at_column ();

create table public.booking_analytics (
analytics_id uuid not null default extensions.uuid_generate_v4 (),
date date not null,
total_bookings integer null default 0,
confirmed_bookings integer null default 0,
cancelled_bookings integer null default 0,
total_revenue numeric(10, 2) null default 0,
new_users integer null default 0,
returning_users integer null default 0,
average_booking_value numeric(10, 2) null,
occupancy_rate numeric(5, 2) null,
created_at timestamp with time zone null default now(),
constraint booking_analytics_pkey primary key (analytics_id)
) TABLESPACE pg_default;

create index IF not exists idx_booking_analytics_date on public.booking_analytics using btree (date) TABLESPACE pg_default;

create table public.booking_slots (
slot_id uuid not null default extensions.uuid_generate_v4 (),
slot_name character varying(100) not null,
start_time time without time zone not null,
end_time time without time zone not null,
duration_hours integer not null,
is_weekday_only boolean null default false,
base_rate numeric(10, 2) not null,
is_active boolean null default true,
downpayment_amount numeric null,
cancellation_fee_amount numeric null,
constraint booking_slots_pkey primary key (slot_id)
) TABLESPACE pg_default;

create table public.bookings (
booking_id uuid not null default extensions.uuid_generate_v4 (),
user_id uuid null,
guest_id uuid null,
check_in_date date not null,
check_in_time time without time zone null,
check_out_date date not null,
check_out_time time without time zone null,
booking_slot_id uuid null,
google_calendar_event_id character varying(255) null,
google_calendar_link text null,
status character varying(50) not null default 'pending'::character varying,
total_amount numeric(10, 2) not null,
down_payment_amount numeric(10, 2) null,
client_questions text null,
number_of_guests integer null,
created_at timestamp with time zone null default now(),
updated_at timestamp with time zone null default now(),
confirmed_at timestamp with time zone null,
cancelled_at timestamp with time zone null,
access_token text not null,
booking_remarks text null,
constraint bookings_pkey primary key (booking_id),
constraint bookings_access_token_key unique (access_token),
constraint bookings_booking_slot_id_fkey foreign KEY (booking_slot_id) references booking_slots (slot_id),
constraint bookings_guest_id_fkey foreign KEY (guest_id) references guest_users (guest_id) on delete set null,
constraint check_user_or_guest check (
(
(
(user_id is not null)
and (guest_id is null)
)
or (
(user_id is null)
and (guest_id is not null)
)
)
)
) TABLESPACE pg_default;

create index IF not exists idx_bookings_created_at on public.bookings using btree (created_at) TABLESPACE pg_default;

create index IF not exists idx_bookings_user_id on public.bookings using btree (user_id) TABLESPACE pg_default;

create index IF not exists idx_bookings_guest_id on public.bookings using btree (guest_id) TABLESPACE pg_default;

create index IF not exists idx_bookings_status on public.bookings using btree (status) TABLESPACE pg_default;

create index IF not exists idx_bookings_check_in_date on public.bookings using btree (check_in_date) TABLESPACE pg_default;

create trigger calculate_booking_down_payment BEFORE INSERT
or
update on bookings for EACH row
execute FUNCTION calculate_down_payment ();

create trigger update_bookings_updated_at BEFORE
update on bookings for EACH row
execute FUNCTION update_updated_at_column ();

create table public.cancellations (
cancellation_id uuid not null default extensions.uuid_generate_v4 (),
booking_id uuid null,
reason text not null,
cancelled_by_user_id uuid null,
cancelled_by_guest_id uuid null,
cancellation_fee numeric(10, 2) null default 2000.00,
cancellation_fee_paid boolean null default false,
waiver_signed boolean null default false,
waiver_document_url text null,
refund_amount numeric(10, 2) null,
refund_status character varying(50) null default 'pending'::character varying,
created_at timestamp with time zone null default now(),
processed_at timestamp with time zone null,
processed_by uuid null,
status character varying null,
access_token text not null,
cancellation_remarks text null,
constraint cancellations_pkey primary key (cancellation_id),
constraint cancellations_booking_id_fkey foreign KEY (booking_id) references bookings (booking_id) on delete CASCADE,
constraint cancellations_cancelled_by_guest_id_fkey foreign KEY (cancelled_by_guest_id) references guest_users (guest_id),
constraint cancellations_cancelled_by_user_id_fkey foreign KEY (cancelled_by_user_id) references users (user_id),
constraint cancellations_processed_by_fkey foreign KEY (processed_by) references admins (admin_id),
constraint check_cancelled_by_user_or_guest check (
(
(
(cancelled_by_user_id is not null)
and (cancelled_by_guest_id is null)
)
or (
(cancelled_by_user_id is null)
and (cancelled_by_guest_id is not null)
)
)
)
) TABLESPACE pg_default;

create table public.comments (
comment_id uuid not null default extensions.uuid_generate_v4 (),
booking_id uuid null,
user_id uuid null,
guest_id uuid null,
comment_text text not null,
rating integer null,
sentiment_score numeric(3, 2) null,
sentiment_label character varying(20) null,
is_visible_to_owner boolean null default true,
is_visible_to_public boolean null default false,
admin_response text null,
responded_by uuid null,
responded_at timestamp with time zone null,
created_at timestamp with time zone null default now(),
constraint comments_pkey primary key (comment_id),
constraint comments_booking_id_fkey foreign KEY (booking_id) references bookings (booking_id) on delete CASCADE,
constraint comments_guest_id_fkey foreign KEY (guest_id) references guest_users (guest_id),
constraint comments_responded_by_fkey foreign KEY (responded_by) references admins (admin_id),
constraint comments_user_id_fkey foreign KEY (user_id) references users (user_id),
constraint check_comment_user_or_guest check (
(
(
(user_id is not null)
and (guest_id is null)
)
or (
(user_id is null)
and (guest_id is not null)
)
)
),
constraint comments_rating_check check (
(
(rating >= 1)
and (rating <= 5)
)
)
) TABLESPACE pg_default;

create table public.guest_users (
guest_id uuid not null default extensions.uuid_generate_v4 (),
first_name character varying(100) not null,
last_name character varying(100) not null,
email character varying(255) not null,
phone_number character varying(20) not null,
created_at timestamp with time zone null default now(),
id_proof_gdrive_url text null,
id_proof_gdrive_link_id text null,
supabase_id_proof_url text null,
supabase_id_proof_path text null,
constraint guest_users_pkey primary key (guest_id)
) TABLESPACE pg_default;

create table public.guidelines (
guideline_id uuid not null default extensions.uuid_generate_v4 (),
category character varying(50) not null,
title character varying(255) null,
content text not null,
display_order integer null,
is_active boolean null default true,
created_at timestamp with time zone null default now(),
updated_at timestamp with time zone null default now(),
updated_by uuid null,
constraint guidelines_pkey primary key (guideline_id),
constraint guidelines_updated_by_fkey foreign KEY (updated_by) references admins (admin_id)
) TABLESPACE pg_default;

create table public.notifications (
notification_id uuid not null default extensions.uuid_generate_v4 (),
user_id uuid null,
admin_id uuid null,
notification_type character varying(50) not null,
title character varying(255) not null,
message text not null,
booking_id uuid null,
payment_id uuid null,
is_read boolean null default false,
read_at timestamp with time zone null,
created_at timestamp with time zone null default now(),
constraint notifications_pkey primary key (notification_id),
constraint notifications_admin_id_fkey foreign KEY (admin_id) references admins (admin_id),
constraint notifications_booking_id_fkey foreign KEY (booking_id) references bookings (booking_id),
constraint notifications_payment_id_fkey foreign KEY (payment_id) references payments (payment_id),
constraint notifications_user_id_fkey foreign KEY (user_id) references users (user_id)
) TABLESPACE pg_default;

create table public.payments (
payment_id uuid not null default extensions.uuid_generate_v4 (),
booking_id uuid null,
reference_number character varying(100) not null,
sender_name character varying(255) not null,
account_number character varying(100) null,
email character varying(255) null,
amount numeric(10, 2) not null,
payment_type character varying(50) not null,
payment_proof_url text null,
payment_proof_storage_path text null,
payment_status character varying(50) null default 'pending'::character varying,
verified_by uuid null,
verified_at timestamp with time zone null,
created_at timestamp with time zone null default now(),
updated_at timestamp with time zone null default now(),
supabase_payment_proof_url text null,
supabase_payment_proof_path text null,
constraint payments_pkey primary key (payment_id),
constraint payments_reference_number_key unique (reference_number),
constraint payments_booking_id_fkey foreign KEY (booking_id) references bookings (booking_id) on delete CASCADE,
constraint payments_verified_by_fkey foreign KEY (verified_by) references admins (admin_id)
) TABLESPACE pg_default;

create index IF not exists idx_payments_booking_id on public.payments using btree (booking_id) TABLESPACE pg_default;

create index IF not exists idx_payments_reference_number on public.payments using btree (reference_number) TABLESPACE pg_default;

create index IF not exists idx_payments_status on public.payments using btree (payment_status) TABLESPACE pg_default;

create trigger update_payments_updated_at BEFORE
update on payments for EACH row
execute FUNCTION update_updated_at_column ();

create table public.users (
user_id uuid not null default extensions.uuid_generate_v4 (),
username character varying(50) not null,
email character varying(255) not null,
password_hash character varying(255) not null,
first_name character varying(100) null,
last_name character varying(100) null,
phone_number character varying(20) null,
created_at timestamp with time zone null default now(),
updated_at timestamp with time zone null default now(),
is_active boolean null default true,
id_proof_gdrive_url text null,
id_proof_gdrive_link_id text null,
supabase_id_proof_url text null,
supabase_id_proof_path text null,
constraint users_pkey primary key (user_id),
constraint users_email_key unique (email),
constraint users_username_key unique (username)
) TABLESPACE pg_default;

create index IF not exists idx_users_email on public.users using btree (email) TABLESPACE pg_default;

create index IF not exists idx_users_username on public.users using btree (username) TABLESPACE pg_default;

create trigger update_users_updated_at BEFORE
update on users for EACH row
execute FUNCTION update_updated_at_column ();

# Instructions to Strictly follow:

### Admin Functionalities:

- Admin Base HTML:
    - make the sidebar with the following routes: Dashboard, Notifications, Booking, Analytics, Edit Profile/Profile Page
- Admin Login Page:
    - Make it the same looking ui but if we’re to login on the admin route, it should have a different color as the user login:
    
    ![image.png](attachment:64a08a36-f04f-4d91-898e-49a81055aaca:image.png)
    
- Admin Dashboard/Home route:
    - turn it into the main content wherein it includes the Calendar(linked to the Google Calendar) and shows the pending, booked and cancelled dates in the calendar
- Admin Notifications Page:
    - Make sure that the Admin will receive an email about the new booking or updated things, the email is stored here: os.environ.get('GMAIL_SMTP_USERNAME')
    - The Admin notifications page consists of the two components which are the admin_notification.html which contains the table where it shows the bookings that are pending, cancelled and confirmed, and admin_notif_detailed.html shows the detailed view of a selected booking and where you should put a functionality where the admin can update whether to confirm, pending and cancel the bookings. (note: there’s an existing logic on these)
    - Take note that there are code in these, you just need to polish things and to properly add things to the respective databases affected, properly send the email update to the customer and the admin, and fix the status_link being sent.
    - Also make sure the bookings and cancellation requests are being sent to admin accounts so they can make sure to have a pending, confirmed and cancelled categories for each type of requests.
    - Add a part where the users and guest users whose bookings did not get approved may reupload their ID photo(if this is a problem for their booking as per admin’s review) or pay the downpayment they haven’t paid yet(if the payment is incomplete or if they didnt pay the downpayment admin’s review or haven’t upload a screenshot of their downpayment)
- Admin Booking Page:
    - On the Admin Bookings page, it’s where the admins get to see the current bookings and past history of bookings
    - Add a page wherein the Admins can book a customer on the website if ever they need to book someone and follow the existing booking logic I have in the guest_booking logics, but instead turn it into something where the Admins will book for the customer.
- Admin Profile Page:
    - Make the Admin Profile page wherein they can see their data and update their data

### Logged in User Functionalities:

- Cancellation Logic:
    - Fix the cancellation logic and especially the part where the status link for their cancellation will be sent on their email again.
    - Polish things that needs to be polished in this part of the project
- Booking:
    - Polish all of the things that are needed to be polished without breaking the existing code in here
    - Make sure that the appropriate tables for the database is being used on this
    - Add a reschedule logic too for the bookings on the status_link sent to their emails aside from the cancellation logic
- User profile page:
    - The user may edit and see their profile in their dedicated profile page
    - Add a part where the user can see all of the bookings and the booking history he did

### Guest User Functionalities:

- Booking
    - Add a prompt to the guest user after they successfully filled out a booking request where if they want to make an account for the website, they may and their booking should be linked on the new account they created if they created one, else, they remain guest_users and the only thing they can do is to use the status_link to check their booking
    - Add a reschedule logic too for the bookings on the status_link sent to their emails aside from the cancellation logic

### Overall website:

- Make a template hero page for the website
- Create new routes if needed and use the existing routes to get context clues on how I am supposed to code everything.
- Make sure all the javascripts are in their dedicated javascript files to clean everything up
- Make the css styles for all of the pages including the inline css for the email templates that will be sent on the email
- Use appropriate javascript for things that needs it like those dropdown arrows on the house rules in the booking page
- Make an about page where we can put the package & inclusions, House Rules, how to book, how to cancel, and how to reschedule
- Contact Us page where we can put the socials and the numbers etc.
- Fix the UI using this style on the photo:

![image.png](attachment:b59c711c-e2b9-426e-8646-441dd1ef0cd7:image.png)

Lastly, make a [summary.md](http://summary.md) files containing all of the changes you’ve made on the whole project and what we need to know and explain all of the code