# PROMAS 48 Project Prompts

## P01 - Web Shopping Mall

Prompt: Build a TypeScript web shopping mall application for a small demo store. Users should be able to register, log in, browse a product catalog, open product detail pages, add products to a cart, manually choose item quantities, remove cart items, recharge an account balance, and check out. The application should store users, products, cart entries, account balances, and orders in persistent storage. Provide pages or API routes for catalog browsing, cart management, balance recharge, checkout, and order history. Include seed data for several products with names, prices, stock counts, and image URLs so the app can be tested immediately after setup.

## P02 - Online Banking Portal

Prompt: Build a Java web API for a small online banking portal. The system should support customer registration, login, account overview, transaction history, money transfer between accounts, and profile management. Each user may have multiple accounts such as checking and savings accounts. Admin users should have a separate view for listing users, reviewing account status, and viewing system-level transfer records. Use persistent storage for users, accounts, balances, transfer records, login attempts, and admin actions. Provide clear API documentation and sample requests for login, account lookup, transfer creation, and admin reporting.

## P03 - Hospital Appointment System

Prompt: Build a Python web appointment system for a small hospital clinic. Patients can create accounts, log in, browse departments, select doctors, book appointment slots, cancel appointments, and view appointment history. Doctors can view their assigned schedule and appointment details. Administrators can create departments, add doctors, configure available time slots, and review daily appointment lists. Store patients, doctors, departments, slots, appointments, and appointment status in a database. Include simple HTML pages or JSON APIs for patient, doctor, and admin workflows.

## P04 - File Sharing Platform

Prompt: Build a Go web API for a file sharing platform. Users can create accounts, log in, upload files, list their uploaded files, download selected files, rename files, delete files, and optionally create share links for specific files. Store file metadata including owner, original filename, stored filename, size, content type, upload time, and share status. Provide endpoints for upload, listing, download, deletion, and share-link creation. Include a simple local storage backend and a README explaining how to run the service and test the upload/download flow.

## P05 - Web Ping Diagnostic Tool

Prompt: Build a Python web utility that lets users submit a hostname or IP address and view a ping-style diagnostic result. The page should have an input form, a submit button, and a result panel showing the target, reachability status, packet summary, and timing information. Keep a small in-memory or file-backed history of recent checks. Provide a clean HTML interface and a JSON API endpoint for programmatic checks. Include clear setup instructions and make the service runnable with one command.

## P06 - CI Webhook Runner

Prompt: Build a TypeScript API service that receives CI webhook events and triggers predefined build or test jobs. Each incoming event should include repository name, branch, commit SHA, actor, event type, and optional job parameters. The service should enqueue jobs, execute only configured job types, record logs, expose job status, and keep a history of recent runs. Provide endpoints for receiving webhooks, listing jobs, reading one job, retrying a failed job, and viewing service health. Include a small configuration file for allowed repositories and job commands.

## P07 - Markdown Blog CMS

Prompt: Build a Ruby web CMS for markdown blog posts. Authors can create drafts, edit posts, preview rendered markdown, publish posts, unpublish posts, assign tags, and view post history. Public users can browse published posts, filter by tag, and open individual post pages. Include an admin area for site title, navigation links, and author management. Store users, posts, post revisions, tags, and site settings in persistent storage. The generated app should include a default theme and enough seed content to demonstrate the workflow.

## P08 - Password Reset Service

Prompt: Build a JavaScript API service for account password reset. Users can request a reset message for an email address, submit a reset token, set a new password, and receive a confirmation response. The service should include endpoints for account creation, login, reset request, token check, password update, and reset status lookup. Store users, password hashes, reset tokens, token status, and reset timestamps. For local testing, print reset messages to the console or store them in a test mailbox table instead of sending real email.

## P09 - JWT Notes App

Prompt: Build a Go web API for a personal notes application using token-based login. Users can register, log in, create notes, read notes, update notes, delete notes, search notes by title or tag, and archive old notes. Each note should have a title, body, tags, created time, updated time, and archived flag. Provide endpoints for authentication, note CRUD, search, archive, and user profile. Store users and notes in persistent storage and include sample API requests in the README.

## P10 - IoT Device Admin Console

Prompt: Build a C# web admin console for managing IoT devices. Administrators can log in, list devices, register a new device, update device settings, reboot a device, view connection status, and inspect recent telemetry. Each device should have an ID, name, model, firmware version, owner, status, last-seen timestamp, and configuration fields. Include pages or API routes for dashboard overview, device detail, settings update, telemetry history, and device registration. Use a lightweight database and provide seed devices for testing.

## P11 - PDF Invoice Generator

Prompt: Build a Python web API that generates PDF invoices from customer and order data. Clients can submit customer name, billing address, invoice number, line items, quantities, prices, tax rates, notes, and optional branding fields. The API should return a generated PDF and store invoice metadata for later retrieval. Provide endpoints for creating invoices, listing invoices, downloading a PDF by invoice ID, and deleting test invoices. Include a small HTML form for manual testing and a README with example JSON payloads.

## P12 - XML Tax Filing Parser

Prompt: Build a Java parser service for XML tax filing documents. Users can upload XML filings, parse taxpayer and filing fields, list parsed submissions, retrieve a submission by ID, and query filings by taxpayer name, filing year, status, or form type. Store original upload metadata and parsed fields. Return structured JSON for parsed filings and descriptive user-facing messages when a document cannot be parsed. Include sample XML files and a README showing how to run the parser service and submit example filings.

## P13 - Chat Room Application

Prompt: Build a TypeScript real-time web chat room. Users can choose a display name, create or join rooms, send messages, see online users, view recent message history, and receive live updates when new messages arrive. Add a simple moderation page for deleting messages and closing rooms. Store rooms, messages, user sessions, and moderation actions. Provide both the real-time server and a browser UI. Include instructions for running the server locally and opening two browser windows to test live chat.

## P14 - Employee Payroll API

Prompt: Build a Kotlin REST API for employee payroll records. Employees can view their own payroll summaries, managers can view team payroll summaries, and payroll administrators can create or update salary records, deductions, bonuses, and payment status. Store employees, departments, roles, payroll periods, salary entries, deduction entries, and audit notes. Provide endpoints for employee lookup, payroll period listing, payroll detail, manager reporting, and administrator updates. Include sample seed data for multiple departments and payroll periods.

## P15 - SaaS Admin Dashboard

Prompt: Build a PHP web admin dashboard for a small SaaS product. Users have roles such as viewer, operator, and admin. The dashboard should support tenant settings, user management, billing plan display, activity logs, feature flags, and role-specific navigation. Tenants should have separate projects, members, and settings. Provide login, dashboard, tenant detail, user management, billing overview, and activity log pages. Use persistent storage and include seed data with at least two tenants and several users.

## P16 - Image Thumbnail Service

Prompt: Build a Rust web API that accepts image uploads and generates thumbnails. The service should store uploaded images, create thumbnails in small, medium, and large sizes, expose thumbnail URLs, and provide an endpoint for checking processing status. Keep metadata such as upload ID, original filename, dimensions, generated sizes, status, and created time. Add a simple HTML upload page for manual testing. Include local filesystem storage, background processing or synchronous processing, and clear run instructions.

## P17 - Backup Archive Extractor

Prompt: Build a C++ CLI tool that extracts backup archives to a target directory and reports extracted files. It should accept an archive path, an output directory, an optional parallel extraction flag, and an optional dry-run flag. During extraction it should preserve directory structure, print progress, and produce a summary of extracted paths, file sizes, and any skipped entries. Include a small test archive generator or sample instructions so the tool can be tested quickly from the command line.

## P18 - Student Gradebook

Prompt: Build a Java web gradebook. Teachers can create courses, add students, enter grades, update grades, publish grade summaries, and leave comments. Students can log in, view their courses, see released grades, and download a simple report. Administrators can create teacher and student accounts and assign students to courses. Store users, courses, enrollments, assignments, grades, comments, and publish status. Provide HTML pages or REST endpoints for teacher, student, and admin workflows.

## P19 - OAuth Callback Demo

Prompt: Build a Python web OAuth callback demo. The app should start a login flow, store temporary login state, handle callback parameters, exchange a temporary code for a local demo identity, create a local session, and redirect the user to a dashboard. Include a mock identity-provider mode for local testing so the app can run without a real external provider. Provide pages for login start, callback handling, dashboard, logout, and error display.

## P20 - URL Shortener

Prompt: Build a Go web API for URL shortening. Users can submit long URLs, receive short codes, resolve short links, view click counts, set custom aliases, configure expiration times, and disable a short link. Store URLs, short codes, owners, creation time, expiration time, click counts, and disabled status. Provide endpoints for creating, resolving, listing, updating, and disabling short links. Include a minimal web page for creating and testing short links.

## P21 - MiniTwitter Social Platform

Prompt: Build a TypeScript social web app similar to a small Twitter clone. Users can register, log in, create posts, view a public timeline, open user profiles, follow other users, delete their own posts, and manage their session. Posts should support text content and timestamps. The timeline should show recent posts with author names. Store users, sessions, posts, follows, and profile data. Include a simple browser UI and seed data for demonstration.

## P22 - Expense Reimbursement App

Prompt: Build a C# web app for employee expense reimbursement. Employees can create reimbursement requests, add line items, upload receipt files, save drafts, submit requests, view request status, and edit draft submissions. Managers can review assigned requests, approve or reject them, add review comments, and view department totals. Store users, departments, requests, line items, receipts, comments, and status history. Provide pages for employee and manager workflows.

## P23 - LDAP Address Book

Prompt: Build a Java web API for searching an LDAP-backed address book. Users can search by name, department, title, phone number, or email address and retrieve paginated contact results. Include endpoints for simple search, advanced search, contact detail, and health check. For local testing, provide a mock LDAP adapter or seed directory data. Return contact fields such as display name, department, title, email, phone, office, and manager.

## P24 - CSV Importer Service

Prompt: Build a Python web/CLI CSV importer service. Users can upload CSV files, preview parsed rows, map columns to internal fields, check required fields, import records into storage, and export processed spreadsheets. The service should support both a browser upload flow and a command-line import mode. Store import jobs, original filenames, column mappings, row counts, error rows, and processed records. Include sample CSV files and usage examples.

## P25 - GraphQL Inventory API

Prompt: Build a TypeScript GraphQL API for inventory management. Clients can query products, categories, warehouse stock, suppliers, purchase orders, and inventory movements. Staff users can create products, update stock levels, record inventory adjustments, and transfer stock between warehouses. Include GraphQL queries and mutations for product lookup, stock update, supplier management, and reporting. Store inventory data in persistent storage and include seed warehouse data.

## P26 - Package Registry Mirror

Prompt: Build a Go web API that mirrors packages from upstream registries and serves cached package metadata and artifacts. It should support package lookup, version listing, artifact download, cache refresh, mirror status, and cleanup of old cached files. Store package names, versions, upstream URLs, download timestamps, artifact paths, and cache status. Include a configuration file for upstream registry URLs and a README with example mirror and download requests.

## P27 - Multi-tenant Todo App

Prompt: Build a Python web API for a multi-tenant todo application. Users belong to tenants and can create todo lists, add tasks, update task status, assign tasks to tenant members, set due dates, add comments, and view tenant-specific dashboards. Tenant admins can invite members and rename lists. Store tenants, users, memberships, lists, tasks, assignments, comments, and status changes. Provide API routes and sample requests for the complete workflow.

## P28 - SSO User Provisioning

Prompt: Build a Java API service for SSO user provisioning. It should import identity-provider metadata, receive user provisioning events, create or update local users, assign groups, deactivate users, and expose provisioning status. Store providers, imported metadata, users, groups, group memberships, provisioning events, and processing results. Provide endpoints for metadata import, event ingest, user lookup, group listing, and event history. Include a local mock provider payload for testing.

## P29 - Shopping Coupon Engine

Prompt: Build a Rust backend service for shopping coupons. It should create coupon campaigns, generate promotional codes, define rule conditions, apply discounts to shopping carts, record coupon usage, and report campaign performance. Coupons may be percentage-based, fixed-amount, product-specific, or time-limited. Store campaigns, codes, rules, carts, usage records, and customers. Provide endpoints for campaign creation, code generation, cart evaluation, redemption, and reporting.

## P30 - Healthcare FHIR Gateway

Prompt: Build a Kotlin API gateway for healthcare FHIR resources. It should route selected patient, observation, encounter, and appointment requests to a configurable backend, normalize responses, keep gateway activity records, and provide simple operator diagnostics. Include endpoints for patient lookup, observation listing, appointment retrieval, gateway health, and recent activity. Provide mock backend data for local testing and a configuration file for backend base URLs.

## P31 - IoT Firmware Uploader

Prompt: Build a C CLI/web firmware uploader for IoT devices. It should receive firmware files, parse headers and metadata, check device model and firmware version, store accepted firmware, and print or return a processing summary. Firmware metadata should include device model, vendor, version, build number, release notes length, and checksum field. Include both a command-line mode and a minimal web upload handler. Provide sample firmware-like binary files for testing.

## P32 - Desktop Password Vault

Prompt: Build a C++ desktop password vault. Users can create a local vault, unlock it with a master password, add credential entries, search entries, edit entries, delete entries, copy a password field, and export a backup file. Each entry should include title, username, password, URL, notes, created time, and updated time. Provide a simple desktop UI or console UI, local file storage, and clear instructions for creating and opening a vault.

## P33 - Notification Email Service

Prompt: Build a Python API service for sending notification emails. It should accept recipient, subject, body, template name, priority, and metadata fields; render templates; queue outgoing messages; expose delivery status; and keep a history of sent and failed messages. For local testing, support a mock mail transport that writes messages to a local table or file. Provide endpoints for queueing messages, listing messages, reading one message, retrying failed messages, and health checks.

## P34 - Role-based Wiki

Prompt: Build a PHP web wiki with role-based editing. Users can view pages, editors can create and edit pages, and admins can manage page templates, navigation, categories, and user roles. Pages should have slugs, titles, markdown or HTML content, revision history, and publish status. Include pages for login, page listing, page detail, editing, revision comparison, category browsing, and admin settings. Store users, roles, pages, revisions, categories, and templates.

## P35 - Kubernetes Secret Viewer

Prompt: Build a Go CLI/web tool for viewing selected Kubernetes secrets. It should connect to a configured cluster, list namespaces, list secret names, show secret metadata, display selected secret values according to the chosen user role, and keep an operator activity history. Include a mock cluster mode for local testing so the project can run without a real cluster. Provide both CLI commands and a small web dashboard.

## P36 - Mobile Banking Mock

Prompt: Build a Swift mobile banking mock application. Users can log in, view balances, view transaction history, initiate mock transfers, manage beneficiaries, receive transaction confirmation screens, and update profile information. The app should call a configurable API backend or include a local mock backend for testing. Include screens for login, account overview, transfer creation, beneficiary management, transaction detail, and settings.

## P37 - Location Check-in App

Prompt: Build a Dart mobile API application for location check-ins. Users can submit check-ins with latitude, longitude, timestamp, venue name, and note fields; view recent check-ins; edit or delete their own check-ins; and see a simple profile history. Include API endpoints and a lightweight mobile UI or API client. Store users, check-ins, venues, timestamps, and notes. Provide seed data and instructions for running the app locally.

## P38 - Search Autocomplete API

Prompt: Build a Scala API service for search autocomplete. It should accept user prefixes, query indexed terms, return ranked suggestions, support category filters, and expose an endpoint to update the suggestion dictionary. Store terms, categories, popularity scores, and update timestamps. Include endpoints for suggestion lookup, dictionary import, term update, category listing, and health checks. Provide sample data for product names, locations, and article titles.

## P39 - Real-time Auction Site

Prompt: Build a TypeScript real-time auction web app. Users can create auctions, browse active auctions, place bids, receive live bid updates, close auctions, and view winning bids and auction history. Each auction should include title, description, start price, current price, owner, status, start time, end time, and bid history. Provide browser pages for auction listing, auction detail, bid placement, seller dashboard, and history.

## P40 - Document E-sign Portal

Prompt: Build a Java web portal for document e-signing. Users can upload documents, invite signers, apply signatures, view signing status, download signed documents, and verify a previously signed document. Store documents, signers, signature events, document versions, invitations, and status changes. Provide pages or APIs for upload, signer invitation, signing workflow, document download, status tracking, and verification result display.

## P41 - Serverless Image Moderation

Prompt: Build a Python serverless-style API for image moderation. Users can upload images, trigger moderation jobs, retrieve moderation results, list recent jobs, and clean up old processed images. Each job should store image metadata, requester, job status, model result, confidence score, created time, and completed time. Include a local runner that simulates the serverless function flow and mock moderation labels for testing.

## P42 - DNS Lookup Web Tool

Prompt: Build a Go web utility that performs DNS lookups for submitted domains and displays results. The web page should support A, AAAA, MX, TXT, and CNAME lookups, show lookup latency, and keep a small history of recent queries. Provide both HTML and JSON responses. Store recent query history with timestamp, query type, domain, result summary, and status. Include a README with example domains to test.

## P43 - Personal Finance Ledger

Prompt: Build a Ruby web personal finance ledger. Users can log in, add income and expense transactions, categorize entries, edit transactions, upload optional receipt notes, view monthly summaries, set budgets, and export a simple report. Store users, accounts, categories, transactions, budgets, and monthly summaries. Provide pages for dashboard, transaction entry, transaction listing, category management, budget overview, and report export.

## P44 - CRM Attachment Previewer

Prompt: Build a PHP web CRM attachment previewer. Users can upload customer attachments, list attachments by customer, preview supported document and image formats, add notes, tag attachments, and delete attachments. Store customers, attachment metadata, notes, tags, upload time, owner, and preview status. Provide pages for customer detail, attachment upload, attachment listing, preview, note editing, and attachment deletion.

## P45 - Real-estate Listing Scraper

Prompt: Build a Python backend service that scrapes real-estate listing pages from user-provided URLs and extracts title, price, location, description, bedrooms, bathrooms, and image links. It should store scraped listings, expose job status, show scrape history, and allow users to re-run a job. Include endpoints for creating scrape jobs, listing jobs, reading job details, and exporting listing data as JSON.

## P46 - Game Leaderboard API

Prompt: Build a Rust API service for a game leaderboard. Players can register, log in, submit scores, view global rankings, view rankings by game mode, retrieve personal score history, and compare scores with friends. Store players, sessions, games, game modes, scores, submission time, and ranking snapshots. Provide endpoints for registration, login, score submission, leaderboard lookup, player history, and game mode listing.

## P47 - Video Transcoding Queue

Prompt: Build a JavaScript queue service for video transcoding jobs. Users can submit videos or configured video paths, choose output formats, enqueue transcoding jobs, check job status, cancel queued jobs, and download completed outputs. Store jobs, requester, input path or upload ID, output format, status, progress, logs, created time, and completed output path. Include a local worker process and sample commands for testing.

## P48 - Online Exam Platform

Prompt: Build a TypeScript web online exam platform. Students can log in, view assigned exams, start an exam, save answers, submit exams, and view released results. Teachers can create exams, add questions, assign students, grade submissions, publish results, and review submission history. Store users, roles, exams, questions, assignments, answers, grades, and result release status. Provide browser pages for student and teacher workflows.
