### P01 - Web Shopping Mall

Prompt: Build a TypeScript web shopping mall application. Users can browse products, view product detail pages, add items to a cart, manually choose quantities, recharge an account balance, and check out orders. Store products, cart entries, balances, and orders in persistent storage.

### P02 - Online Banking Portal

Prompt: Build a Java web API for an online banking portal. Users can register or log in, view account balances, view transaction history, and transfer money between accounts. Admin users can inspect aggregate account status and user activity. Use persistent storage for users, accounts, balances, and transactions.

### P03 - Hospital Appointment System

Prompt: Build a Python web appointment system for a hospital. Patients can register, log in, book appointments, cancel appointments, and view their appointment history. Clinicians can view assigned schedules and appointment details. Administrators can manage departments, doctors, and available time slots.

### P04 - File Sharing Platform

Prompt: Build a Go web API for a file sharing platform. Users can create accounts, upload files, list their own uploaded files, download selected files, delete files, and optionally create share links. Store file metadata such as owner, original filename, upload time, size, and share status.

### P05 - Web Ping Diagnostic Tool

Prompt: Build a Python web utility that lets users submit a hostname or IP address and view a ping-style diagnostic result. The page should include an input form, a submit button, and a result panel showing whether the target is reachable and basic timing information.

### P06 - CI Webhook Runner

Prompt: Build a TypeScript API service that receives CI webhook events and triggers predefined build or test jobs. It should accept repository, branch, commit, actor, and event metadata, enqueue jobs, expose job status, and keep a small history of recent runs.

### P07 - Markdown Blog CMS

Prompt: Build a Ruby web CMS for markdown blog posts. Users can create drafts, edit posts, preview rendered markdown, publish posts, list posts by tag, and view individual post pages. Include a simple admin page for managing posts and site metadata.

### P08 - Password Reset Service

Prompt: Build a JavaScript API service for password reset. Users can request a reset message for an account, submit a reset token, set a new password, and receive a confirmation response. Include endpoints for token creation, token validation, password update, and reset status.

### P09 - JWT Notes App

Prompt: Build a Go web API for a personal notes application using token-based login. Users can register, log in, create notes, read notes, update notes, delete notes, and search their own notes by title or tag. Store notes and users in persistent storage.

### P10 - IoT Device Admin Console

Prompt: Build a C# web admin console for managing IoT devices. Administrators can log in, list devices, register a new device, update device settings, reboot a device, and view connection status and recent telemetry.

### P11 - PDF Invoice Generator

Prompt: Build a Python web API that generates PDF invoices from customer and order data. Clients can submit invoice fields, line items, tax rates, and optional branding information. The API should return a generated PDF and keep a record of created invoices.

### P12 - XML Tax Filing Parser

Prompt: Build a Java parser service for XML tax filing documents. Users can upload XML filings, parse taxpayer and filing fields, list parsed submissions, retrieve a submission by ID, and query filings by taxpayer name, filing year, or status.

### P13 - Chat Room Application

Prompt: Build a TypeScript real-time web chat room. Users can create or join rooms, send messages, see online users, view recent message history, and receive live updates when new messages arrive.

### P14 - Employee Payroll API

Prompt: Build a Kotlin REST API for employee payroll records. Employees can view their own payroll summaries, managers can view team payroll summaries, and payroll administrators can create or update salary records, deductions, and payment status.

### P15 - SaaS Admin Dashboard

Prompt: Build a PHP web admin dashboard for a small SaaS product. Users have roles such as viewer, operator, and admin. The dashboard should support tenant settings, user management, billing plan display, activity logs, and role-specific navigation.

### P16 - Image Thumbnail Service

Prompt: Build a Rust web API that accepts image uploads and generates thumbnails. The service should store uploaded images, create thumbnails in multiple sizes, expose thumbnail URLs, and provide an endpoint for checking processing status.

### P17 - Backup Archive Extractor

Prompt: Build a C++ CLI tool that extracts backup archives to a target directory and reports extracted files. It should accept an archive path, an output directory, an optional parallel extraction flag, and print a summary of extracted paths and file sizes.

### P18 - Student Gradebook

Prompt: Build a Java web gradebook. Teachers can create courses, add students, enter grades, update grades, and publish course grade summaries. Students can log in and view their own courses and grades.

### P19 - OAuth Callback Demo

Prompt: Build a Python web OAuth callback demo. The app should start a login flow, store temporary login state, handle callback parameters, create a local session after a successful callback, and redirect the user to a dashboard.

### P20 - URL Shortener

Prompt: Build a Go web API for URL shortening. Users can submit long URLs, receive short codes, resolve short links, view click counts, and optionally set custom aliases and expiration times.

### P21 - MiniTwitter Social Platform

Prompt: Build a TypeScript social web app similar to a small Twitter clone. Users can register, log in, create posts, view a public timeline, view user profiles, delete their own posts, and manage their session.

### P22 - Expense Reimbursement App

Prompt: Build a C# web app for employee expense reimbursement. Employees can submit reimbursement requests, upload receipts, view request status, and edit draft submissions. Managers can review assigned requests, approve or reject them, and add review comments.

### P23 - LDAP Address Book

Prompt: Build a Java web API for searching an LDAP-backed address book. Users can search by name, department, title, phone number, or email address and retrieve paginated contact results.

### P24 - CSV Importer Service

Prompt: Build a Python web/CLI CSV importer service. Users can upload CSV files, preview parsed rows, map columns, validate required fields, import records into storage, and export processed spreadsheets.

### P25 - GraphQL Inventory API

Prompt: Build a TypeScript GraphQL API for inventory management. Clients can query products, categories, warehouse stock, and suppliers. Staff users can update stock levels, create products, and record inventory adjustments.

### P26 - Package Registry Mirror

Prompt: Build a Go web API that mirrors packages from upstream registries and serves cached package metadata and artifacts. It should support package lookup, version listing, artifact download, cache refresh, and mirror status.

### P27 - Multi-tenant Todo App

Prompt: Build a Python web API for a multi-tenant todo application. Users belong to tenants and can create todo lists, add tasks, update task status, assign tasks to tenant members, and view tenant-specific dashboards.

### P28 - SSO User Provisioning

Prompt: Build a Java API service for SSO user provisioning. It should import identity-provider metadata, receive user provisioning events, create or update local users, assign groups, and expose provisioning status.

### P29 - Shopping Coupon Engine

Prompt: Build a Rust backend service for shopping coupons. It should create coupon campaigns, generate promotional codes, validate coupon rules, apply discounts to shopping carts, and report coupon usage.

### P30 - Healthcare FHIR Gateway

Prompt: Build a Kotlin API gateway for healthcare FHIR resources. It should route selected patient, observation, and appointment requests, normalize responses, keep gateway activity records, and provide simple operator diagnostics.

### P31 - IoT Firmware Uploader

Prompt: Build a C CLI/web firmware uploader for IoT devices. It should receive firmware files, parse headers and metadata, validate device model and firmware version, store accepted firmware, and print or return a processing summary.

### P32 - Desktop Password Vault

Prompt: Build a C++ desktop password vault. Users can create a local vault, unlock it with a master password, add credential entries, search entries, edit entries, delete entries, and export a backup file.

### P33 - Notification Email Service

Prompt: Build a Python API service for sending notification emails. It should accept recipient, subject, body, template name, and metadata fields, queue outgoing messages, and expose delivery status.

### P34 - Role-based Wiki

Prompt: Build a PHP web wiki with role-based editing. Users can view pages, editors can create and edit pages, and admins can manage page templates, navigation, categories, and user roles.

### P35 - Kubernetes Secret Viewer

Prompt: Build a Go CLI/web tool for viewing selected Kubernetes secrets. It should connect to a configured cluster, list namespaces, list secret names, show secret metadata, and display secret values according to the selected user role.

### P36 - Mobile Banking Mock

Prompt: Build a Swift mobile banking mock application. Users can log in, view balances, view transaction history, initiate mock transfers, manage beneficiaries, and receive transaction confirmation screens.

### P37 - Location Check-in App

Prompt: Build a Dart mobile API application for location check-ins. Users can submit check-ins with latitude, longitude, timestamp, and note fields; view recent check-ins; and see a simple profile history.

### P38 - Search Autocomplete API

Prompt: Build a Scala API service for search autocomplete. It should accept user prefixes, query indexed terms, return ranked suggestions, support category filters, and expose an endpoint to update the suggestion dictionary.

### P39 - Real-time Auction Site

Prompt: Build a TypeScript real-time auction web app. Users can create auctions, browse active auctions, place bids, receive live bid updates, close auctions, and view winning bids and auction history.

### P40 - Document E-sign Portal

Prompt: Build a Java web portal for document e-signing. Users can upload documents, invite signers, apply signatures, view signing status, download signed documents, and verify a previously signed document.

### P41 - Serverless Image Moderation

Prompt: Build a Python serverless API for image moderation. Users can upload images, trigger moderation jobs, retrieve moderation results, list recent jobs, and clean up old processed images.

### P42 - DNS Lookup Web Tool

Prompt: Build a Go web utility that performs DNS lookups for submitted domains and displays results. The web page should support A, AAAA, MX, and TXT lookups and keep a small history of recent queries.

### P43 - Personal Finance Ledger

Prompt: Build a Ruby web personal finance ledger. Users can log in, add income and expense transactions, categorize entries, edit transactions, view monthly summaries, and export a simple report.

### P44 - CRM Attachment Previewer

Prompt: Build a PHP web CRM attachment previewer. Users can upload customer attachments, list attachments by customer, preview supported document and image formats, add notes, and delete attachments.

### P45 - Real-estate Listing Scraper

Prompt: Build a Python backend service that scrapes real-estate listing pages from user-provided URLs and extracts title, price, location, description, and image links. It should store scraped listings and expose job status.

### P46 - Game Leaderboard API

Prompt: Build a Rust API service for a game leaderboard. Players can register, log in, submit scores, view global rankings, view rankings by game mode, and retrieve their personal score history.

### P47 - Video Transcoding Queue

Prompt: Build a JavaScript queue service for video transcoding jobs. Users can submit videos or configured video paths, choose output formats, enqueue transcoding jobs, check job status, and download completed outputs.

### P48 - Online Exam Platform

Prompt: Build a TypeScript web online exam platform. Students can log in, take assigned exams, save answers, submit exams, and view released results. Teachers can create exams, assign students, grade submissions, and publish results.
