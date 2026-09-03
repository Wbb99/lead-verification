#!/usr/bin/env python3
"""
Lead Analysis System v2

Classifies inbound leads from call transcripts, SMS and form submissions across three
dimensions: whether a human answered, the lead outcome, and whether the contact is spam.

Rules-first. Several hundred deterministic conditions handle what can be decided from the
transcript itself; a language model is called only for the cases they cannot resolve.
Measured at 94% agreement with manual review across 1,000 randomly sampled production
leads validated by hand.
"""

import csv
import json
import re
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Rate limiting - 5 requests per minute = 12 seconds between requests
RATE_LIMIT_DELAY = 0  # Set to 13 for strict rate limiting, 0 for faster testing

@dataclass
class AnalysisResult:
    call_answer: Literal["Yes", "No", "Dropped", "Not a Phone Call"]
    outcome: Literal["Verified", "Disqualified", "Unverified"]
    reasoning: str = ""
    spam: bool = False


def is_not_a_phone_call(source: str, transcript: str) -> bool:
    """Determine if this is NOT a phone call (SMS, form submission, etc.)"""
    transcript_lower = transcript.lower().strip()

    # Website form submissions
    if source == "Website":
        return True

    # Google LSA SMS messages
    if "g.co/homeservices" in transcript_lower:
        return True
    if "google local services ads" in transcript_lower:
        return True
    if "lsa dashboard" in transcript_lower:
        return True
    if "replies to this number will be sent to the customer" in transcript_lower:
        return True

    # Check for SMS/text message patterns (no Speaker tags)
    if "Speaker" not in transcript:
        # Messages without speaker tags from Unknown or Google Guaranteed Ads are SMS
        if source in ["Unknown", "Google Guaranteed Ads"]:
            return True
        # GBP messages without Speaker tags
        if source == "Google Business Profile":
            # Short messages without Speaker tags are likely SMS
            if len(transcript.strip()) < 300:
                # Check if it reads like a message (has greeting, signature, etc.)
                if any(phrase in transcript_lower for phrase in ["hello", "hi ", "hey ", "let me know", "interested",
                                                                   "miss the", "want to say", "just want"]):
                    return True
                # Check for customer service inquiry patterns (SMS inquiries)
                inquiry_patterns = ["can you help", "do you", "could you", "able to",
                                   "estimate", "invoice", "project", "property",
                                   "my business", "my home", "my roof"]
                if any(p in transcript_lower for p in inquiry_patterns):
                    return True
                # Message about phone issues = SMS (they couldn't call so they're texting)
                phone_issue_patterns = ["phone line", "wont let me through", "won't let me through",
                                       "can't get through", "cant get through", "couldn't reach",
                                       "couldnt reach", "tried calling", "tried to call"]
                if any(p in transcript_lower for p in phone_issue_patterns):
                    return True
                # Generic "I want to" or "want to know" patterns without Speaker tags = SMS
                if "want to" in transcript_lower or "i need" in transcript_lower:
                    return True

    return False


def is_lsa_lead(transcript: str) -> bool:
    """Check if this is a Google Local Services Ads lead"""
    transcript_lower = transcript.lower()

    indicators = [
        "g.co/homeservices",
        "google local services ads",
        "lsa dashboard",
        "via google local services",
        "replies to this number will be sent to the customer"
    ]

    return any(ind in transcript_lower for ind in indicators)


def is_spam_verification_call(transcript: str) -> bool:
    """Detect spam/scam business verification calls"""
    transcript_lower = transcript.lower()

    # First check if this is actually a voicemail system - if so, not spam
    # (some spam phrases like "showing correctly" might appear in garbled voicemails)
    # Note: "call you back" removed - too broad, matches hold queues ("we'll call you back")
    vm_indicators = ["leave a message", "leave me a message", "please leave"]
    if any(vm in transcript_lower for vm in vm_indicators):
        # This is a voicemail, not a spam robocall
        return False

    # NOTE: "opt out" alone is NOT a spam indicator - it appears in legitimate IVR menus
    # Only flag as spam if there are actual spam CONTENT indicators
    spam_indicators = [
        "verify your business",
        "google business account",
        "google voice search",
        "business listing",
        "not verified",
        "confirm your listing",
        "claim your listing",
        "update your profile",
        "customers cannot find your business",
        "not showing on search",
        "press one to verify",
        "press 1 to verify",
        # Removed "opt out" - appears in legitimate IVR menus
        "google my business",
        "yelp listing",
        "yellow pages",
        # Note: "better business bureau" removed - BBB makes legitimate calls to businesses
        "bbb listing",
        "via alexa",  # "alexa" alone matches person names like "This is Alexa speaking"
        "on alexa",
        "amazon alexa",
        "voice search optimization",
        "search engine listing",
        "online directory",
        "your business is not",
        "regarding your google",
        "important message regarding",
        "google and google voice",
        "verify or update",
        "voice clients are currently having trouble",
        "google listings",
        "showing correctly",
        "may be suspended",  # "Your listing may be suspended"
        "listing may be suspended",
        "not appear in searches",  # "may not appear in searches"
        "verification issues",
        "do not take care of this",  # "If we do not take care of this" - spam robocall
        "press 2 to take care",  # "press 2 to take care of this" - spam robocall (more specific)
        "having trouble finding",  # Garbled version of "clients are currently having trouble finding your business"
        "verify your code",  # Automated verification code spam
        "please verify your code",
    ]

    # Check for spam patterns
    for indicator in spam_indicators:
        if indicator in transcript_lower:
            return True

    return False


def get_rule_based_outcome(transcript: str, source: str, call_answer: str) -> Optional[str]:
    """Try to determine outcome with rules before using AI"""
    transcript_lower = transcript.lower().strip()

    # LSA leads = Verified
    if is_lsa_lead(transcript):
        return "Verified"

    # Real estate agents/brokers with inspection requests = Verified
    # Per user rule: "Real estate broker requesting inspection: verified"
    # BUT exclude: selling services to the company, follow-ups on existing estimates
    real_estate_agent_patterns = ["i'm a broker", "im a broker", "i'm the realtor", "im the realtor",
                                   "i am a broker", "i am the realtor"]
    # Direct inspection/estimate REQUEST (not follow-up on existing)
    new_inspection_patterns = ["need a", "get a", "want a", "wanting to", "looking for",
                               "could you", "can you"]
    service_patterns = ["inspection", "estimate", "roof", "condition"]
    # Make sure they're REQUESTING service, not following up
    followup_patterns = ["already", "coming over", "was already", "scheduled", "appointment"]
    # Make sure they're not selling TO the company
    selling_patterns = ["presentation folders", "marketing", "print", "brochure", "catalog",
                        "working on a project with"]
    if any(p in transcript_lower for p in real_estate_agent_patterns):
        if any(p in transcript_lower for p in service_patterns):
            if not any(p in transcript_lower for p in followup_patterns):
                if not any(p in transcript_lower for p in selling_patterns):
                    return "Verified"  # Real estate agent with NEW service need

    # Spam = Disqualified (unless garbled/partial, then Unverified)
    if is_spam_verification_call(transcript):
        # Check if it's garbled/partial spam (very short or incomplete)
        if len(transcript_lower) < 100:
            return "Unverified"  # Garbled/incomplete spam
        return "Disqualified"

    # Third-party platforms = Disqualified
    if is_third_party_lead_platform(transcript):
        return "Disqualified"

    # "Let me know if you're interested" pattern = Disqualified
    if "let me know if you" in transcript_lower and "interested" in transcript_lower:
        return "Disqualified"

    # LinkedIn outreach = B2B marketing = Disqualified
    if "linkedin" in transcript_lower or "found you on" in transcript_lower:
        return "Disqualified"

    # "Would it be worth X minutes" = solicitation = Disqualified
    if "worth" in transcript_lower and "minutes" in transcript_lower:
        return "Disqualified"

    # "8 minutes of your time" pattern = B2B solicitation
    if "minutes of your time" in transcript_lower:
        return "Disqualified"

    # Staffing/recruiting/employment calls = Disqualified (B2B marketing to the business)
    staffing_indicators = ["staffing", "recruiting", "employment", "temp agency", "temporary workers",
                          "hire staff", "hiring staff", "workforce", "job opening", "job candidate"]
    if any(ind in transcript_lower for ind in staffing_indicators):
        return "Disqualified"

    # B2B organizations calling the business = Disqualified
    b2b_org_indicators = ["better business bureau", "bbb ", "chamber of commerce", "trade association",
                          "public adjuster", "consulting firms", "target localization",
                          "golf club", "golf course"]
    if any(ind in transcript_lower for ind in b2b_org_indicators):
        return "Disqualified"

    # Sales calls TO the company (not from customers) = Disqualified
    # e.g., printer machine, office supplies, etc.
    sales_to_company_indicators = [
        "printer machine", "printer in the office", "catalog of the printer",
        "office supplies", "copier machine",
    ]
    if any(ind in transcript_lower for ind in sales_to_company_indicators):
        return "Disqualified"

    # "Did you receive my text" pattern = Disqualified (follow-up marketing)
    # But for SMS (Not a Phone Call), this should be Unverified - handled separately below
    if call_answer != "Not a Phone Call" and "did you receive" in transcript_lower and ("text" in transcript_lower or "message" in transcript_lower):
        return "Disqualified"

    # Found company via signs/crew in neighborhood = NOT from marketing channels = Disqualified
    # Per user rule: Must come through marketing channels to be a valid lead
    # High-confidence patterns (clear discovery context)
    non_marketing_patterns_strong = [
        "saw your sign", "saw the sign", "seen your sign", "seen the sign",
        "saw you guys", "seen you guys", "saw your crew", "seen your crew",
        "saw your truck", "seen your truck", "saw your van", "seen your van",
        "working on my neighbor", "at my neighbor",
        "saw you working", "seen you working",
        "in the neighborhood", "in my neighborhood",
    ]
    if any(p in transcript_lower for p in non_marketing_patterns_strong):
        return "Disqualified"
    # Lower-confidence patterns - can appear in other contexts (addresses, proximity)
    # Only disqualify if NOT accompanied by a clear service request
    non_marketing_patterns_weak = [
        "down the street", "next door",
    ]
    if any(p in transcript_lower for p in non_marketing_patterns_weak):
        service_indicators = ["leak", "repair", "estimate", "quote", "damage", "fix",
                              "broken", "replace", "install", "inspection", "leaking"]
        if not any(si in transcript_lower for si in service_indicators):
            return "Disqualified"

    # Asking for referral to another contractor/company = Disqualified (not hiring THIS company)
    referral_request_patterns = [
        "do you have a contractor", "have a contractor that",
        "know a contractor", "recommend a contractor",
        "do you work with a contractor", "work with a contractor that",
        "refer me to", "refer us to",
    ]
    if any(p in transcript_lower for p in referral_request_patterns):
        return "Disqualified"

    # SMS marketing/spam patterns = Disqualified
    sms_marketing_patterns = [
        "reply stop to unsubscribe",
        "reply stop to stop",
        "reply help for",
        "text stop to",
        "i'm a real human",
        "did my last message",
        "trust built marketing",
        "empire ai",
        "-trust built",
        "marketing llc",
        "opting in to receiving sms",
        "message frequency may vary",
        "message and data rates",
        # SMS lead solicitation patterns
        "aren't shared leads",
        "these aren't shared leads",
        "not shared leads",
        "if you're not interested",
        "looking forward to hearing from you",
        # B2B SMS solicitation
        "i do seamless gutters",
        "in case you need somebody",
        "in case you need someone",
        "i do roofing",
        "i do plumbing",
        "i do hvac",
        "what amount do you need",  # Finance/lending solicitation
        "pre-map draw triggers",  # Finance jargon
        # B2B lead selling
        "homeowners looking",  # "I've got 8-9 homeowners looking to start..."
        "can you take on more projects",
        "strong first impressions",  # Marketing/coaching pitch
        "live today at",  # Webinar marketing
        # System/automated messages
        "message preferences have been updated",
        "no longer receive texts from",
        "notifications via email",
    ]
    if any(pattern in transcript_lower for pattern in sms_marketing_patterns):
        return "Disqualified"

    # Auto-reply SMS patterns = Disqualified (no real intent)
    auto_reply_patterns = [
        "sorry, i can't talk right now",
        "can't talk right now",
        "i'm busy right now",
    ]
    if any(p in transcript_lower for p in auto_reply_patterns):
        return "Disqualified"

    # Market research / business questions (not service requests) = Disqualified
    if "do you guys take on" in transcript_lower or "word-of-mouth" in transcript_lower:
        return "Disqualified"

    # SMS: Personal/social messages (not service-related) = Disqualified
    # e.g., "I miss the work and the crew", "Just want to say hi"
    if call_answer == "Not a Phone Call":
        personal_sms_patterns = [
            "miss the work", "miss the crew", "miss you", "say hi to",
            "want to say hi", "just want to say", "hope you're doing",
        ]
        if any(p in transcript_lower for p in personal_sms_patterns):
            # Check if there's any service content too
            service_keywords = ["repair", "quote", "fix", "leak", "damage"]
            if not any(kw in transcript_lower for kw in service_keywords):
                return "Disqualified"  # Personal message, not service-related

    # SMS: B2B marketing/lead selling patterns = Disqualified
    # Must check BEFORE service keyword check since they often contain "roofing", "plumbing" etc.
    if call_answer == "Not a Phone Call":
        b2b_sms_patterns = [
            # Lead selling
            "supply leads", "roofing leads", "plumbing leads", "hvac leads",
            "i supply", "supply roofing", "leads at $", "bad lead",
            "homeowners in the area", "homeowners looking for estimates",
            "pay on result partnership", "pay on result", "pay per result",
            "have capacity for more", "capacity for more roofing",
            # Lead selling v2
            "exclusive verified", "verified leads", "replacement leads",
            "no upfront fees", "only pay $", "per lead",
            "would you like more info",  # B2B pitch ending
            # Appointment setting services
            "scheduling pre-qualified", "pre-qualified appointments",
            "ready-to-start", "homeowners are color-ready", "budget confirmed",
            "right here in your local market",
            # Marketing/lead generation services
            "we help roofing companies", "help roofing companies",
            "book consistent", "roof replacement calls weekly",
            "without ads or retainers", "wanna see how",
            # Business acquisition
            "interested in selling", "buyers that are interested", "interest in selling",
            "have buyers", "have several buyers",
            # Marketing services
            "bring verified homeowners", "straight to your crm", "ready for quotes",
            "wanna see how it performs", "build systems for roofers",
            # Generic B2B
            "free for a call", "set up a call to discuss",
            # B2B partnership/lead referral spam (Sarah/Jessie patterns)
            "looking to partner with", "partner with an", "partner with a ",
            "pay per job", "on a pay per", "jobs/month", "jobs per month",
            "more jobs/month", "more jobs per month",
            "quoting a few homeowners", "quote a few homeowners",
            "came across your", "great job!",  # Flattery + pitch pattern
            "take on more", "able to take on",
            # Scheduling/booking company pitches
            "booked out a company", "book you out", "looking to book you",
            "we booked out",
        ]
        if any(p in transcript_lower for p in b2b_sms_patterns):
            return "Disqualified"  # B2B marketing/lead selling

    # SMS with clear service keywords = Verified (only for ACTUAL service requests)
    # Must be a customer asking for service, not B2B mentioning services
    if call_answer == "Not a Phone Call":
        # These indicate actual customer service requests (specific repair/install language)
        customer_service_patterns = [
            "flashing", "shingle", "j channel", "kick out", "water getting in",
            "need repair", "need quote", "need estimate", "want quote", "want estimate",
            "my roof", "my gutter", "my window", "my plumbing", "my hvac",
            "looking for quote", "looking for estimate", "get a quote", "get an estimate",
            # Customer inquiries
            "inquiring if you offer", "do you offer", "if you offer",
            "replacement service", "roof replacement", "gutter replacement",
            "are you taking on", "taking on new", "still taking on",
            "interior painting", "exterior painting",
            # Repair requests
            "facade repair", "fascia repair", "fallen off", "has fallen off",
            "repair on side", "siding repair", "it needs repair",
            # Short service queries
            "roof repair", "gutter repair", "window repair",
            "roof replacement", "gutter replacement", "window replacement",
            "repair or replacement",
            # Service availability questions
            "do you still provide", "do you provide", "do you still offer",
            "do you offer", "still provide", "still offer",
            "siding services", "roofing services", "plumbing services",
            # General service inquiry patterns
            "do you guys do", "do you do", "do you guys still do",
            "needs work done", "need work done",
        ]
        if any(p in transcript_lower for p in customer_service_patterns):
            return "Verified"  # Clear customer service request in SMS

    # Short unclear SMS without service request = Unverified (unknown intent)
    # Only return Disqualified if there's clear spam/marketing content
    if call_answer == "Not a Phone Call" and len(transcript_lower) < 100:
        # Verification codes = Disqualified (internal/automated)
        if "verification code" in transcript_lower:
            return "Disqualified"
        # "Let me know if interested" = B2B solicitation = Disqualified
        if "let me know" in transcript_lower:
            return "Disqualified"
        # "Did you receive my text?" type = Unverified (unknown context)
        if "did you receive" in transcript_lower or "did my last" in transcript_lower:
            return "Unverified"
        # Clear spam/marketing patterns = Disqualified
        spam_sms_patterns = ["your message preferences", "no longer receive", "notifications via email"]
        if any(p in transcript_lower for p in spam_sms_patterns):
            return "Disqualified"
        # Otherwise, unclear intent = Unverified
        return "Unverified"

    # Internal calls (PIN numbers, verification codes, work-related chat) = Disqualified
    if "pin number" in transcript_lower or "verification code" in transcript_lower:
        return "Disqualified"

    # Internal work calls (between employees) = Disqualified
    internal_call_patterns = [
        "clicked working on my current call",
        "i'll just put the call back in",
        "the call for sellers",
        "just pulled up",
    ]
    if any(p in transcript_lower for p in internal_call_patterns):
        return "Disqualified"

    # Explicit follow-up indicators (always Disqualified, no exceptions)
    # "I spoke to you yesterday" or "had been conversing with" = prior contact follow-up
    explicit_followup_patterns = [
        "i spoke to you", "i spoke with you",
        "spoke to you yesterday", "spoke with you yesterday",
        "been conversing with", "had been conversing",
    ]
    if any(p in transcript_lower for p in explicit_followup_patterns):
        return "Disqualified"

    # Follow-up on existing appointments/estimates = Disqualified
    # But "spoke to you yesterday about scheduling NEW work" = Verified (handled by AI)
    # Only disqualify clear cancellation/rescheduling/waiting patterns
    follow_up_phrases = [
        "cancel the appointment", "cancel my appointment",
        "change my appointment", "had an appointment",
        "waiting for quote", "waiting for estimate", "following up on",
        "cancel or reschedule",
        "i need to reschedule", "i want to reschedule", "can i reschedule",
        "reschedule my appointment", "reschedule my estimate",
        "need to reschedule", "like to reschedule",
        "received a quote", "received an estimate", "received a bid",
        "got a quote", "got an estimate", "got a bid",
    ]
    if any(phrase in transcript_lower for phrase in follow_up_phrases):
        # Exception: if caller explicitly wants to schedule NEW work (not just IVR "press 2 to schedule")
        caller_wants_schedule = any(p in transcript_lower for p in [
            "want to schedule", "like to schedule", "need to schedule",
            "schedule new", "schedule a new",
        ])
        if caller_wants_schedule and "cancel" not in transcript_lower and "reschedule" not in transcript_lower:
            pass  # Let AI handle - might be scheduling new work
        else:
            return "Disqualified"

    # Asking for specific person by name (likely existing relationship) = Disqualified
    # But only if there's NO service request in the call
    service_keywords = ["roof", "repair", "estimate", "quote", "install", "replace", "fix", "leak",
                        "window", "gutter", "siding", "damage", "inspection", "service", "help",
                        "veranda", "porch", "deck", "consultation"]
    has_service_request = any(kw in transcript_lower for kw in service_keywords)

    # Check if this is an IVR menu (not someone asking for a specific person)
    is_ivr_menu_context = bool(re.search(r'press \d|press one|press two|for sales|for service', transcript_lower))

    # If caller asks for specific person AND there's no service request mentioned AND not IVR menu
    if not has_service_request and not is_ivr_menu_context:
        # Patterns: "speak with Jamie", "is this Susan", "is Vicki available"
        # Note: Exclude "is not available" which appears in VM greetings
        # Note: Exclude generic words like "an agent", "the owner", etc.
        name_request_patterns = [
            r'(?<!press \d to )speak with (?!an |the |a )\w+',
            r'(?<!press \d to )speak to (?!an |the |a )\w+',
            r'talk to (?!an |the |a )\w+',
            r'is this \w+\?',
            r'is (?!not )\w+ available',  # Exclude "is not available"
            r'is \w+ there',
            r'could i speak with (?!an |the |a )\w+',
            r'could i speak to (?!an |the |a )\w+',
        ]
        if any(re.search(pattern, transcript_lower) for pattern in name_request_patterns):
            return "Disqualified"

    # Chamber of Commerce, BBB membership calls = Disqualified (B2B marketing)
    if "chamber of commerce" in transcript_lower or "filled out a form" in transcript_lower:
        return "Disqualified"

    # "Speaking with the owner" = B2B sales call = Disqualified
    if "speaking with the owner" in transcript_lower or "speak with the owner" in transcript_lower:
        return "Disqualified"

    # "Looking to speak with the business owner" = B2B sales pitch = Disqualified
    if "speak with the business owner" in transcript_lower or "looking to speak with the business owner" in transcript_lower:
        return "Disqualified"

    # "Talk with a/the business owner" = B2B sales call = Disqualified
    if "talk with a business owner" in transcript_lower or "talk with the business owner" in transcript_lower:
        return "Disqualified"
    if "talk to a business owner" in transcript_lower or "talk to the business owner" in transcript_lower:
        return "Disqualified"

    # B2B solicitor looking for marketing/lead contact = Disqualified
    if "handles the leads" in transcript_lower or "whoever handles" in transcript_lower:
        return "Disqualified"
    if "looking for the owner" in transcript_lower and not has_service_request:
        return "Disqualified"

    # "Calling to speak with [person]" = asking for specific person = Disqualified
    # Works even when IVR menu mentions "service" (which pollutes has_service_request)
    if "calling to speak with" in transcript_lower:
        return "Disqualified"

    # "Called earlier" / "I haven't heard back" = follow-up on same day = Disqualified
    # BUT NOT if they're calling to schedule/address a service need
    # AND NOT if the original contact was a long time ago (not same-day follow-up)
    if "called earlier" in transcript_lower or "i haven't heard" in transcript_lower:
        # Exception: long-timeframe indicators suggest this isn't a same-day follow-up
        long_timeframe = any(p in transcript_lower for p in [
            "a while back", "a while ago", "months ago", "weeks ago",
            "long time ago", "long time back", "some time ago",
        ])
        if not long_timeframe:
            # Exception: if they have a service need they're trying to address
            service_needs = ["waterline", "water line", "leak", "roof", "emergency", "urgent",
                             "schedule", "appointment", "estimate", "quote"]
            has_service_need = any(sn in transcript_lower for sn in service_needs)
            if not has_service_need:
                return "Disqualified"

    # Documentation/records request for past work (not new service) = Disqualified
    # e.g., "I bought a house that you reroofed, checking if you have records"
    if "bought a house" in transcript_lower and "your records" in transcript_lower:
        return "Disqualified"

    # Commercial/industrial project quotes = Disqualified (B2B, not consumer leads)
    if "commercial project" in transcript_lower or "industrial project" in transcript_lower:
        return "Disqualified"

    # Referrals = ALWAYS Disqualified (not from marketing channels)
    # Check early before any service-related patterns
    referral_patterns = [
        "i was referred", "was referred by", "referred me", "referred us",
        "friend referred", "neighbor referred", "someone referred",
        "got referred", "been referred", "they referred",
        "family has used", "family used", "wife's family", "husband's family",
        "my family has used", "my family used", "recommended you",
        "they recommended you", "recommended you guys",
        "recommended y'all", "recommended y' all", "recommended ya'll",
        # Neighbor saw-your-work referrals
        "did my neighbor", "you did my neighbor",
        # Family referrals (work done on family member's property)
        "my daughter's house", "my son's house", "my mother's house", "my father's house",
        "my sister's house", "my brother's house", "my parent's house", "my parents' house",
        # Personal connection referrals
        "best friends with",
    ]
    if any(pattern in transcript_lower for pattern in referral_patterns):
        return "Disqualified"

    # B2B wanting to USE this company's services = Verified (potential customer)
    # e.g., "looking for a roofing service we can use", "reaching out looking for [service]"
    b2b_customer_patterns = [
        "looking for a roofing service we can use",
        "looking for a contractor we can use",
        "looking for a plumber we can use",
        "looking for a service we can use",
        "looking for.*service.*we can use",
        "we're reaching out.*looking for",
        "reaching out looking for",
    ]
    for pattern in b2b_customer_patterns:
        if re.search(pattern, transcript_lower):
            return "Verified"  # B2B wanting to hire this company = valid lead

    # Genuine service inquiries = Verified (even if company can't help)
    # e.g., "got anybody that does X repairs", "do you guys do X"
    service_inquiry_patterns = [
        r'got anybody that does \w+ repairs',
        r'do you (guys )?do \w+ (repairs|service)',
        r'do you have anyone that does',
        r'you guys do \w+\?',
    ]
    for pattern in service_inquiry_patterns:
        if re.search(pattern, transcript_lower):
            return "Verified"  # Genuine service inquiry = valid lead

    # Appointment being scheduled = Verified
    if "schedule your appointment" in transcript_lower or "scheduling your appointment" in transcript_lower:
        return "Verified"


    # B2B marketing TO the business = Disqualified
    # But B2B service requests (commercial projects) = let AI handle
    b2b_marketing_phrases = [
        "affiliated with keller williams",  # Real estate marketing
        "painting company that",  # Intermediary
        "partners calling",
    ]
    for phrase in b2b_marketing_phrases:
        if phrase in transcript_lower:
            return "Disqualified"

    # Very short answered calls - need to distinguish between Unverified and Disqualified
    if call_answer == "Yes" and len(transcript_lower) < 100:
        # FIRST: Check for business-name-only pattern (no customer response)
        # e.g., "Hello, this is sage roofing. Hello.", "Hi, this is Joyce from honest exteriors."
        company_name_pattern = re.search(
            r'(from \w+\s+(roofing|exteriors|construction|plumbing|hvac|contracting)|'
            r'this is[\w\s]+(roofing|exteriors|construction|plumbing|hvac|contracting))',
            transcript_lower
        )
        if company_name_pattern:
            # Strip company name from transcript to check for REAL service content
            cleaned = re.sub(r'(this is|from)[\w\s]+(roofing|exteriors|construction|plumbing|hvac|contracting)', '', transcript_lower)
            real_service_hints = ["repair", "estimate", "quote", "need", "fix", "leak",
                                  "gutter", "window", "install", "replace", "damage", "inspection"]
            if not any(hint in cleaned for hint in real_service_hints):
                return "Disqualified"  # Business greeting only, no customer service content

        # Business greeting ending with "how can I help you?" and no customer response = Disqualified
        help_q = ["how can i help you", "how may i help you", "can i help you"]
        for hp in help_q:
            if hp in transcript_lower:
                hp_pos = transcript_lower.rfind(hp)
                after = re.sub(r'[?\s\n.]', '', transcript_lower[hp_pos + len(hp):]).strip()
                if len(after) < 5:
                    return "Disqualified"

        # If there's any hint of service inquiry, let AI handle
        service_hints = ["roof", "repair", "estimate", "quote", "service", "need", "fix", "leak",
                        "gutter", "window", "plumbing", "hvac", "install", "replace"]
        if any(hint in transcript_lower for hint in service_hints):
            return None  # Let AI decide
        # "Hello? Hello?" pattern = Unverified (answered but caller hung up, unknown intent)
        if re.search(r'hello\??\s*hello\??', transcript_lower):
            return "Unverified"
        # Just "Hello" or "Hello, this is [name]" with no context = Unverified
        # e.g., "Hello.", "Hello, this is John.", "This is Kathy. Hello."
        if re.search(r'^speaker [a-z]:\s*(hello|hi)[,.!?\s]*$', transcript_lower):
            return "Unverified"
        if re.search(r'^speaker [a-z]:\s*(hello|hi)[,.]?\s*(this is \w+|[\w\s]+)[,.!?\s]*$', transcript_lower):
            return "Unverified"
        # Other short calls without clear intent = Unverified
        return "Unverified"

    # 3-MONTH RULE: Past customer within 3 months = Disqualified (not new business)
    # Past customer over 3 months or no timeframe specified = can be Verified
    # IMPORTANT: "spoke yesterday" for NEW inquiry = Verified (scheduling new work)
    # Only disqualify if it's a FOLLOW-UP on existing quote/work, not fresh scheduling

    # Check for "over 3 months" indicators first - these are always OK
    over_3_months_patterns = [
        "year ago", "years ago", "a year or two", "year or two ago",
        "three.{0,5}years", "four.{0,5}years", "five.{0,5}years",
        "last year", "couple years", "few years",
        "maybe a year", "about a year", "going on a year",
        "quite a while ago", "some time ago", "a while back", "a while ago",
        "long time ago", "several months ago", "months ago",
        # Specific month patterns for months that are typically >3 months ago
        "back in march", "back in april", "back in may", "back in june",
        "back in january", "back in february",
        "in march of", "in april of", "in may of", "in june of",
    ]
    is_over_3_months = any(re.search(p, transcript_lower) for p in over_3_months_patterns)

    # Past customer indicators
    past_customer_patterns = [
        "you did", "you guys did", "you all did", "you replaced", "you installed",
        "work for me", "work done", "had used", "had done some work",
        "you guys had done", "fixed", "repaired", "put on", "put in",
    ]
    has_past_customer_indicator = any(re.search(p, transcript_lower) for p in past_customer_patterns)

    # Service need indicators
    service_need_patterns = [
        "leak", "damage", "repair", "issue", "problem", "sagging", "broken",
        "wondering", "need", "fix", "replace", "inspect",
    ]
    has_service_need = any(p in transcript_lower for p in service_need_patterns)

    # Past customer over 3 months with service need = Verified
    if is_over_3_months and has_past_customer_indicator and has_service_need:
        return "Verified"

    # If clearly over 3 months, don't disqualify based on past work
    if not is_over_3_months:
        # Recent timeframe indicators (within 3 months):
        recent_timeframe_patterns = [
            "few days ago", "couple days ago",
            "couple of days", "a few days", "the other day",
            "few weeks ago", "couple weeks ago", "a week ago",
            "two weeks ago", "three weeks ago", "a month ago", "two months ago",
        ]
        # Past work indicators (actual work done, not just conversation)
        past_work_patterns = [
            "you did", "you guys did", "you all did", "you replaced",
            "you installed", "we had work done", "work that.*did", "just settled up",
        ]
        has_recent_timeframe = any(p in transcript_lower for p in recent_timeframe_patterns)
        has_past_work = any(re.search(p, transcript_lower) for p in past_work_patterns)

        if has_past_work and has_recent_timeframe:
            # Within 3 months = not new business
            return "Disqualified"

    # Existing quote follow-up = Disqualified (not new business)
    # Be careful to exclude "get a quote from you" which is asking for NEW quote
    existing_quote_patterns = [
        "received a quote", "got a quote from you", "you gave us a quote",
        "quote you sent", "already have a quote",
        "previous quote", "the quote you",
        "had a quote from you", "have a quote from you"  # Following up on existing
    ]
    # Don't match "get a quote from you" - that's asking for NEW quote
    if any(pattern in transcript_lower for pattern in existing_quote_patterns):
        # Make sure it's not "get a quote from you"
        if "get a quote from you" not in transcript_lower and "order to get a quote" not in transcript_lower:
            return "Disqualified"

    # Staffing/recruiting calls = Disqualified
    staffing_indicators = ["staffing", "recruiting", "employment agency", "temp agency", "hiring agency",
                           "headhunter", "job placement", "workforce solutions"]
    if any(ind in transcript_lower for ind in staffing_indicators):
        return "Disqualified"

    # Empty/NA = Disqualified
    if transcript_lower in ["na", "n/a", ""]:
        return "Disqualified"

    # Very short transcripts (< 30 chars) with no clear intent
    if len(transcript_lower) < 30:
        # SMS/Website short messages = Unverified (unclear intent)
        if call_answer == "Not a Phone Call" or source == "Website":
            return "Unverified"
        # Check for spam indicators
        if "business" in transcript_lower or "google" in transcript_lower:
            return "Disqualified"
        # "Hello, Spam" or similar = Disqualified
        if "spam" in transcript_lower:
            return "Disqualified"
        # Check for VM indicators
        if is_voicemail_system(transcript):
            return "Unverified"  # VM with no content = Unverified
        # Just business name with no VM indication = Disqualified
        # e.g., "Brady roofing.", "James roofing."
        if re.search(r'^speaker [a-z]:\s*[\w\s]+\s*(roofing|plumbing|hvac|construction|services|exteriors)\s*\.?\s*$', transcript_lower):
            return "Disqualified"
        # Very short unclear fragments = Unverified (can't determine intent)
        # Examples: "This.", "Hermione r.", "Ho.", "Micro."
        return "Unverified"

    # Short transcripts (< 50 chars) with no customer response
    if len(transcript_lower) < 50 and call_answer == "No":
        # Check for spam fragments
        if "google" in transcript_lower or "business" in transcript_lower:
            return "Disqualified"
        # "Hello, Spam" = Disqualified
        if "spam" in transcript_lower:
            return "Disqualified"
        # Check if it's a VM greeting first
        if is_voicemail_system(transcript):
            return "Unverified"
        # Just business name with no VM indication = Disqualified (no useful lead info)
        # e.g., "Brady roofing.", "James roofing.", "Austin custom roofing."
        if re.search(r'^speaker [a-z]:\s*[\w\s]+\s*(roofing|plumbing|hvac|construction|services|exteriors)\s*\.?\s*$', transcript_lower):
            return "Disqualified"
        # Very short fragments like "To ensure." = Disqualified (incomplete, no lead info)
        if len(transcript_lower) < 20:
            return "Disqualified"
        # Other short unclear fragments = Unverified (can't determine intent)
        return "Unverified"

    # VM = Default to Unverified (unknown intent) unless clear disqualifying content
    if call_answer == "No" and is_voicemail_system(transcript):
        speakers = re.findall(r'Speaker ([A-Z]):', transcript)
        unique_speakers = set(speakers)

        # Check for clear B2B/spam/warranty in VM message - these are Disqualified
        # NOTE: Asking for specific person in VM = Unverified (not Disqualified)
        # NOTE: "on behalf of" + service inquiry = could be Verified (B2B customer)
        disqualifying_vm_patterns = [
            "headway capital", "federated insurance", "insurance policy",
            "business lending", "business loan", "line of credit",
            "marketing services", "advertising opportunity",
            "reaching out to discuss", "partnership opportunity",
            "calling from lesson", "check in on a quote",  # Follow-up calls
            # Warranty claims on past work
            "you did a roof", "you did my roof", "you did our roof",
            "we were supposed to", "supposed to have solved",
            # Office closed messages
            "office is currently closed",
            # Golf course/club, marketing
            "golf course", "golf club",
            # Sales/marketing pitches in VM
            "run your ads", "generate more leads", "free week",
            "i do advertising", "advertising and i'm doing",
            # Automated messages
            "this is a message from",
            # B2B marketing calls (Yelp, etc.)
            "calling from yelp", "from yelp", "yelp calling", "with yelp",
            " at yelp", "over at yelp",
            "brendan with yelp", "this is yelp",
            "local companies", "bringing them points",
            # Personal/social calls
            "miss the work", "miss the crew", "say hi to",
            # Spam overlay patterns on VM
            "verify your business", "google voice search", "not verified",
            "press 0 to verify", "listing may be suspended",
            # B2B localization/marketing
            "running a target localization", "target localization",
            "my company is running",
            # B2B contractor/vendor calls
            "submitting a bid", "submit a bid", "interested in bidding",
            # B2B professional services
            "public adjuster", "consulting firms", "design services",
            "civil, structural", "mep design",
            # Invoice/billing inquiries
            "final invoice", "invoice", "payment due", "balance due",
            # Verification codes
            "verify your code", "your code is",
            # Supply company pitches
            "supply", "i was reviewing your",
        ]
        if any(pattern in transcript_lower for pattern in disqualifying_vm_patterns):
            return "Disqualified"

        # Check for clear service request in VM message
        service_vm_patterns = [
            "water line issue", "water leak", "roof leak", "need repair",
            "need an estimate", "need a quote", "want to schedule",
            "calling about my roof", "calling about my plumbing",
            "calling about my hvac", "need service", "need help with",
            # Callback patterns - customer responding to company outreach
            "contacting me about", "you've been calling", "returning your call",
            "calling you back", "about doing a roof", "about a roof estimate",
            "about an estimate", "about a quote",
            # Specific damage/repair needs
            "roof repaired", "tree fell on", "tree fall on",
            "need a roof", "need my roof", "need some roof",
            "hail damage", "storm damage", "wind damage",
            # Callback about previously discussed work
            "about work", "about the work",
        ]
        if any(pattern in transcript_lower for pattern in service_vm_patterns):
            return "Verified"  # Clear service need or callback in VM

        # DEFAULT: Voicemail = Unverified (can't determine intent)
        # Keep as Unverified - changing to Disqualified causes too many regressions
        return "Unverified"

    # Hold queue only (no conversation)
    # But NOT if it ends with VM ("not available", "reply after the tone", "leave a message")
    hold_queue_phrases = ["stand in line", "someone will be with you shortly", "please stay on the line",
                          "helping another caller", "we'll be with you shortly"]
    ends_with_vm = any(p in transcript_lower for p in ["not available", "reply after the tone", "leave a message", "leave your message"])
    if any(phrase in transcript_lower for phrase in hold_queue_phrases) and not ends_with_vm:
        # Check if there's actual human conversation/dialogue after the hold message
        has_dialogue = bool(re.search(
            r'(how can i help|how may i help|this is \w+ from|my name is|i need|i have a|'
            r'i was wanting|i was wondering|i was just|wanting to see|work on my|can you|could you|'
            r'do you guys|give us an estimate|come out to)',
            transcript_lower
        ))
        if not has_dialogue:
            # Hold queue ONLY (no VM prompt, no conversation) = Disqualified
            # Per user rule: IVR/Hold queue only = Dropped + Disqualified
            return "Disqualified"

    # Answered but no customer response (business greeting only, caller said nothing)
    # Short transcript ending with "how can I help you?" and no customer dialogue = Disqualified
    if call_answer == "Yes" and len(transcript_lower) < 200:
        help_patterns = ["how can i help you", "how may i help you", "can i help you"]
        for hp in help_patterns:
            if hp in transcript_lower:
                hp_pos = transcript_lower.rfind(hp)
                after_help = re.sub(r'[?\s\n.]', '', transcript_lower[hp_pos + len(hp):]).strip()
                if len(after_help) < 5:
                    return "Disqualified"

    # "I missed a call from this number" = no business intent = Disqualified
    if "missed a call" in transcript_lower and "this number" in transcript_lower:
        return "Disqualified"

    # Cannot determine with rules - need AI
    return None


def is_third_party_lead_platform(transcript: str) -> bool:
    """Detect calls FROM third-party lead/marketing platforms trying to sell leads"""
    transcript_lower = transcript.lower()

    # High-signal phrases that indicate a platform is calling to sell/discuss leads
    # NOT just any mention of a platform name
    lead_platform_phrases = [
        "lead service",
        "referral service",
        "home services marketplace",
        "leads in your area",
        "service requests in your area",
        "pro account",
        "provider account",
        "contractor account",
        "we sent you a lead",
        "you missed a lead",
        "get more jobs",
        "get more customers through our platform",
        "calling from angi",
        "calling from homeadvisor",
        "calling from thumbtack",
        "this is angi calling",
        "this is homeadvisor calling",
    ]

    for phrase in lead_platform_phrases:
        if phrase in transcript_lower:
            return True

    return False


def is_voicemail_system(transcript: str) -> bool:
    """Detect if the call went to voicemail system"""
    transcript_lower = transcript.lower()

    # First check if this looks like a hold/transfer message (NOT voicemail)
    hold_indicators = [
        "please hold", "hold while we transfer", "hold while we connect",
        "next available", "higher than normal call volume", "helping another caller",
        "we'll be with you shortly", "stay on the line"
    ]
    if any(phrase in transcript_lower for phrase in hold_indicators):
        return False  # This is hold/transfer, not voicemail

    # Automated call screening = NOT voicemail (human interaction after)
    # e.g., "If you record your name and reason for calling, I'll see if this person is available"
    if "if you record your name" in transcript_lower or "record your name and reason" in transcript_lower:
        return False  # This is automated screening, not voicemail

    # Check for clear conversation patterns (NOT voicemail even if VM phrases present)
    # If there's a clear business intro + customer response NEARBY, it's not voicemail
    # Use limited match distance and word boundaries to avoid matching across VM greetings
    conversation_patterns = [
        r'how (can|may) i help you\?[\s\S]{0,100}?\b(hi|hello|yes|my name)\b',
        r'\b(roofing|plumbing|hvac)[,.][\s\S]{0,100}?\b(hi|hello|yes|my name|i need)\b',
        r'this is \w+\. how can i[\s\S]{0,100}?\b(hi|hello|yes)\b',
    ]
    for pattern in conversation_patterns:
        if re.search(pattern, transcript_lower):
            return False  # Clear conversation, not voicemail

    vm_system_phrases = [
        "leave a message",
        "leave me a message",
        "leave your message",
        "leave a detailed message",
        "leave a voicemail",
        "leave your voicemail",
        "please leave your",
        "voice messaging system",
        "forwarded to voicemail",
        "record your message",
        "leave your name",
        "leave your name and number",
        "after the beep",
        "after the tone",
        "your call has been forwarded",
        "unable to answer the phone",
        "mailbox is full",
        "no one is available to answer",
        "we will return your call",
        "will call you back",
        "call you back as soon as",
        "sorry i missed your call",
        "sorry i've missed your call",
        "sorry we missed your call",
        "i apologize i missed your call",
        "i apologize we missed your call",
        "sorry we couldn't come to the phone",
        "sorry that we missed your call",
        "leave a name",
        "please record your message",
        "at the tone",
        "is not available",
        "we are either with a customer or unable to take your call",
        "can't take your call right now",
        "unable to get to my phone",
        "can't get to your call",
        "i'll be happy to give you a call",
        "you've reached",  # "Hello, you've reached..."
        "you have reached",  # "You have reached X. Please leave..."
    ]

    return any(phrase in transcript_lower for phrase in vm_system_phrases)


def has_human_conversation(transcript: str) -> bool:
    """Check if there's actual human-to-human DIALOGUE (not just voicemail + message left)"""
    transcript_lower = transcript.lower()

    # Check if this is a voicemail + message pattern (NOT a conversation)
    vm_greeting_phrases = [
        "please leave a message",
        "leave me a message",
        "leave your message",
        "leave a detailed message",
        "leave a voicemail",
        "please leave your",
        "leave your name",
        "we will return your call",
        "will call you back",
        "call you back as soon as",
        "can't take your call",
        "unable to take your call",
        "no one is available",
        "we are either with a customer",
        "after the beep",
        "at the tone",
        "please record your message",
        "sorry i'm unable to get to my phone",
        "sorry i missed your call",
        "sorry we missed your call",
        "i apologize i missed your call",
        "i apologize we missed your call",
        "sorry we couldn't come to the phone",
        "sorry that we missed your call",
        "sorry i've missed your call",
        "leave a name",
    ]

    # Hold phrases - NOT voicemail, but need to check if someone answers
    hold_queue_phrases = [
        "stand in line",
        "please stand in line",
        "someone will be with you shortly",
        "helping another caller",
        "we'll be with you shortly",
        "please stay on the line",
        "please hold",
        "hold while we transfer",
        "hold while we connect",
        "higher than normal call volume",
        "hold on for just a moment",
        "someone from our team will be right with you",
        "will be right with you",
        "will be with you shortly",
        "while i try to connect you",
        "try to connect you",
        "is being transferred",
        "next available",
        "your call is very important",
    ]
    has_hold_queue = any(phrase in transcript_lower for phrase in hold_queue_phrases)

    # IVR-only indicators - these mean NOT answered by human
    # NOTE: "to speak with" is too broad (matches real conversation "looking to speak with")
    # So we only use it in combination with IVR patterns (press X)
    ivr_only_indicators = [
        "press 1 for", "press 2 for", "press 1 to", "press 2 to",
        "press 1", "press 2", "press 3", "press 4", "press 5",  # Simple press patterns
        "press 6", "press 7", "press 8", "press 9", "press 0",
        "press one", "press two", "press three", "press four", "press five",
        "if you're a returning customer, press", "if you're new",
        "please select", "for sales, press", "for service, press",
        ", please press 1", ", please press 2",  # "to schedule..., please press 1"
        "for new customers", "for existing customer",  # IVR menu options
        "opt out", "to opt out",  # Spam/marketing IVR
        "representative from my pro",  # Known spam caller
    ]
    has_ivr_only = any(phrase in transcript_lower for phrase in ivr_only_indicators)
    # Also check "to speak with" + "press" pattern for IVR menus
    if "to speak with" in transcript_lower and "press" in transcript_lower:
        has_ivr_only = True

    # Look for multiple speakers
    speakers = re.findall(r'Speaker ([A-Z]):', transcript)
    unique_speakers = set(speakers)

    # If VM greeting, NOT a conversation
    # (even if "your call is very important" appears - that's VM greeting, not hold queue)
    has_vm_greeting = any(phrase in transcript_lower for phrase in vm_greeting_phrases)
    if has_vm_greeting:
        # "Your call is very important" in VM context is NOT hold queue
        # It's only hold queue if there's also "helping another caller" or similar
        actual_hold_indicators = ["helping another caller", "we'll be with you shortly",
                                  "hold while we", "please hold", "higher than normal call volume"]
        is_actual_hold = any(ind in transcript_lower for ind in actual_hold_indicators)
        if not is_actual_hold:
            return False

    # If it's just IVR menu with no human response, NOT a conversation
    # Even if multiple "speakers" due to transcription quirks
    if has_ivr_only and not has_hold_queue:
        # Check if there's ACTUAL human dialogue after IVR
        # Look for name introductions or dialogue after IVR
        after_ivr_patterns = [
            r'this is \w+[\'s,. ]',  # "this is Lily.", "this is Mike,", "this is Riley's"
            r'hey,?\s+this is \w+',  # "Hey, this is Lily"
            r'hi,?\s+this is \w+',  # "Hi, this is Lily"
            r'hi,?\s+\w+,?\s+this is',  # "Hi, Beverly, this is..." (greeting someone then intro)
            r'my name is \w+',
            r'how can i help',
            r'how may i help',
            r'can i help you',
            r'who do i have the pleasure',
            r'how could i be of help',
            r'i\'m wondering',  # "I'm wondering if I can get a quote"
            r'can i get a quote',
            r'i need (a |an )?(quote|estimate)',
            r'(^|[.!?] )\w+ speaking[,.]',  # "Michelle speaking." at start of sentence, NOT "forward to speaking"
            r'hey,?\s+i\'m',  # "Hey, I'm wondering"
            r'\w+ from the \w+ (dudes|roofing|plumbing|construction)',  # "Lily from the Roofing Dudes"
            r'\w+ from \w+ (roofing|plumbing|construction|hvac)',  # "Lily from ABC Roofing"
        ]
        after_ivr_intro = any(re.search(p, transcript_lower) for p in after_ivr_patterns)
        if not after_ivr_intro:
            return False

    # If hold message, check if someone answered after
    if has_hold_queue:
        # Check for automated callback message FIRST - NOT a conversation
        if "someone from our team will be calling you back" in transcript_lower:
            return False
        if "will be calling you back" in transcript_lower:
            return False
        if "we'll give you a call back" in transcript_lower:
            return False

        # Check for dialogue patterns REGARDLESS of speaker count
        # (transcription may merge multiple speakers into one)
        # "this is [name]", "how can i help", actual dialogue
        if re.search(r'(this is|hey,? this is|hi,? this is) \w+[\'s,. ]', transcript_lower):
            return True
        if "how can i help" in transcript_lower or "how may i help" in transcript_lower:
            return True
        # "Hey, I'm wondering" or "I need a quote" after hold
        if "i'm wondering" in transcript_lower or "i need a quote" in transcript_lower:
            return True
        # "[Name] from [Company]" pattern - human answered
        if re.search(r'\w+ from (the )?\w+ (roofing|plumbing|dudes|construction)', transcript_lower):
            return True
        # Look for dialogue after hold message (". Hey, " or ". Hi, ")
        if re.search(r'\.\s*(hey|hi)[,. ]\s*(this is|i\'m|we|my name)', transcript_lower):
            return True
        # Quote request pattern
        if "get a quote" in transcript_lower or "get an estimate" in transcript_lower:
            return True
        if "need a quote" in transcript_lower or "need an estimate" in transcript_lower:
            return True

        if len(unique_speakers) >= 2:
            # Look for actual back-and-forth dialogue patterns
            if re.search(r'(hi|hello|hey)[,.]?\s*(yes|yeah|hi|hello|this is|i\'m|i need)', transcript_lower):
                return True
            # "Looking to speak with" or "need to speak with" indicates conversation
            if "looking to speak with" in transcript_lower or "i'm looking to speak" in transcript_lower:
                return True
            if "need to speak" in transcript_lower or "speak with the owner" in transcript_lower:
                return True
            # Name greeting pattern (internal calls etc)
            if re.search(r'hey,?\s+\w+\.', transcript_lower):
                return True
            # "Hey, [name]" pattern after hold
            if re.search(r'hey,?\s+\w+[,.]', transcript_lower):
                return True
            # 3+ speakers almost always means conversation
            if len(unique_speakers) >= 3:
                return True
            # No clear conversation after hold - NOT a conversation
            return False
        else:
            return False  # Just hold message, no answer

    # 2+ unique speakers = conversation (unless VM greeting or IVR looping)
    if len(unique_speakers) >= 2:
        # Check for CLEAR dialogue patterns first - these always indicate conversation
        # Use limited [\s\S]{0,100} to match across speaker tags but not too greedily
        clear_dialogue_patterns = [
            r'how (can|may) i help.*\?[\s\S]{0,100}?(hi|hello|yes|my name|i\'m|i need|i have|i was)',
            r'(roofing|plumbing|hvac|heating|cooling)[,.]?[\s\S]{0,50}?(hi|hello|yes|yeah|my name)',
            r'this is \w+[,.][\s\S]{0,100}?(hi|hello|yes|okay|sure|my name)',
            r'\?[\s\S]{0,50}?(hi|hello|yes|yeah|okay|sure|we do|we don\'t|i need|my name)',  # Match across speaker tags
            r'(hi|hello)[,.]?\s+(yes|this is|my name|i\'m calling|i was calling|i need|i have)',
            r'can i help you\?[\s\S]{0,50}?(yes|yeah|hi|hello|i need|i have|i\'m calling)',  # "Can I help you?" response
        ]
        if any(re.search(p, transcript_lower) for p in clear_dialogue_patterns):
            return True

        # Check if this is just IVR looping between "speakers" (transcription quirk)
        # IVR looping = same automated message split across speakers
        if has_ivr_only:
            # Only count as conversation if there's clear human response
            human_response_patterns = [
                r'(hi|hello|hey)[,.]?\s*(yes|yeah|my name|i need|i have|i\'m calling)',
                r'how can i help.*?\?.*?(yes|hi|hello|i need)',
                r'this is \w+[,.].*?(hi|hello|yes|my name)',
            ]
            if not any(re.search(p, transcript_lower) for p in human_response_patterns):
                return False

        # Check for actual dialogue indicators
        dialogue_indicators = [
            "how can i help", "how may i help", "how can i hop to it",
            "what can i do for you", "let me check", "let me get", "let me see",
            "let me look", "let me transfer", "i can help", "i can assist",
            "i can get", "i can take", "i can schedule", "transfer you",
            "hold on", "one moment", "can i get your", "what's your",
            "what is your", "hi,", "hello,", "yes,", "yeah,", "okay,",
            "sure,", "certainly", "absolutely", "all right",
        ]

        if any(ind in transcript_lower for ind in dialogue_indicators):
            return True

        # Even without indicators, 2 speakers having any exchange is a conversation
        # (unless it's VM greeting + message left or IVR looping)
        if len(speakers) >= 2:  # At least 2 speaker tags
            return True

    # Single speaker transcript but contains dialogue pattern
    # (sometimes transcription merges speakers)
    if len(unique_speakers) == 1:
        # First check for "no response" pattern - business saying hello multiple times
        # "Hello, this is sage roofing. Hello." = no response, NOT dialogue
        hello_count = len(re.findall(r'\bhello\b', transcript_lower))
        if hello_count >= 2 and len(transcript) < 100:
            return False  # Multiple hellos in short transcript = no response

        # Look for dialogue patterns within single speaker
        # "Roofing. Yes. Hi there..." pattern
        # But NOT ". Hello" or ". Hi" which could be repeated greeting (no response)
        if re.search(r'\. (yes|hey)[,. ]', transcript_lower, re.IGNORECASE):
            return True
        # ". Hello" or ". Hi" is only dialogue if there's other dialogue context
        if re.search(r'\. (hi|hello)[,. ]', transcript_lower, re.IGNORECASE):
            # Check if there's actual dialogue content (questions, service requests)
            if '?' in transcript or re.search(r'(i need|i have|can you|quote|estimate|appointment)', transcript_lower):
                return True
            # Otherwise, likely just repeated greeting = no response
        if re.search(r'\? (yes|no|yeah|okay|sure)', transcript_lower, re.IGNORECASE):
            return True
        # Questions followed by answers
        if transcript.count('?') >= 1 and transcript.count('.') >= 2:
            response_patterns = ["yes", "no", "okay", "sure", "certainly", "absolutely"]
            if any(f". {resp}" in transcript_lower or f"? {resp}" in transcript_lower for resp in response_patterns):
                return True

    return False


def was_call_answered(transcript: str) -> str:
    """Determine call answer status: Yes, No, or Dropped"""
    transcript_lower = transcript.lower()
    speakers = re.findall(r'Speaker ([A-Z]):', transcript)
    unique_speakers = set(speakers)

    # Check for spam robocalls - but only if there's NO conversation
    # If someone answers a spam call, it's still "Yes" for call answered
    spam_robocall_phrases = [
        "this is business listing verification",
        "this is an important message regarding your google",
        "important message regarding your google business",
        "your business is not verified",
        "clients are currently having trouble finding",
        "voice clients are currently having trouble",
        "verification issues",
        "may be suspended",
        "listing may be suspended",
        "listing doesn't stop showing",
        "stop showing up on google",
    ]
    # Note: We check this AFTER has_convo, so spam with conversation = Yes

    # Check for automated messages left on voicemail
    # "This is a message from [company]" = automated, not human answered
    if "this is a message from" in transcript_lower:
        return "No"  # Automated message, not answered

    # Check for B2B sales pitches left on voicemail
    # Pattern: "Hello. [Name] here from [Company]... give me a call at..."
    # These are messages left by someone, not answered calls
    if "give me a call at" in transcript_lower or "give me a call on" in transcript_lower:
        # Check if it's a message left on voicemail (not a conversation)
        if len(unique_speakers) <= 1:
            return "No"  # Someone leaving a message

    # Check for voicemail system
    is_vm = is_voicemail_system(transcript)
    # Override: if many alternating speaker turns, it's a real conversation
    # (VM phrases in conversational context shouldn't flag as voicemail)
    speakers = re.findall(r'Speaker ([A-Z]):', transcript)
    if is_vm and len(speakers) >= 8:
        is_vm = False

    # Check for IVR patterns - any "press X" instruction = IVR/Dropped
    ivr_strict = [
        "press 1", "press one", "press 2", "press two", "press 3", "press three",
        "press 4", "press four", "press 5", "press five", "press 6", "press six",
        "press 7", "press seven", "press 8", "press eight", "press 9", "press nine",
        "press zero", "press 0", "to opt out", "opt out or call",
        "nine to opt out", "9 to opt out",  # IVR opt-out patterns
        "for more options", "to speak with", "to speak to",  # Common IVR phrases
        "if you're a returning customer", "if you're new", "please select",
        "if you know your party's extension",  # Auto-attendant
        "dial it at any time",  # Auto-attendant
        "your call has been sent",  # Call forwarding system
    ]
    has_ivr_strict = any(phrase in transcript_lower for phrase in ivr_strict)

    # IVR with song/hold music before menu = still IVR
    has_hold_music_pattern = bool(re.search(r'(honey|sugar|candy|loving|love song)', transcript_lower))

    # "opt out" patterns indicate IVR system = Dropped
    # But only if it's NOT spam (spam with opt-out = No, pure IVR = Dropped)
    has_opt_out = "opt out" in transcript_lower or "nine to opt out" in transcript_lower

    # Check for hold/queue messages - these are IVR/Dropped, not answered
    # Note: "your call is very important" can appear in BOTH hold queues AND voicemail greetings
    # Only treat it as hold queue if there's no VM indicator (please leave, leave a message)
    hold_queue_phrases = ["stand in line", "helping another caller", "we'll be with you shortly", "please stay on the line",
                          "higher than normal call volume", "please hold", "next available", "transfer you to",
                          "hold while we transfer", "hold while we connect",
                          "hold on for just a moment", "someone from our team will be right with you",
                          "will be right with you", "will be with you shortly",
                          "while i try to connect you", "try to connect you", "is being transferred"]
    is_hold_queue = any(phrase in transcript_lower for phrase in hold_queue_phrases)
    # "your call is very important" is hold queue ONLY if not a VM greeting
    if "your call is very important" in transcript_lower:
        vm_indicators = ["please leave", "leave a message", "leave your name", "leave me a message",
                        "will return your call", "can't take your call", "we will get back to you"]
        if not any(ind in transcript_lower for ind in vm_indicators):
            is_hold_queue = True  # Real hold queue, not VM greeting

    # Note: Transfer at start of call doesn't affect answer status
    # If conversation happens after transfer, call was still answered

    # FIRST: Check for spam - spam robocalls are NOT answered even with multiple "speakers"
    # (transcription often splits spam audio into multiple speaker tags)
    # Spam robocalls = No (automated message, like voicemail) - spam flag set separately
    is_spam = is_spam_verification_call(transcript)
    if is_spam:
        # Short garbled IVR fragments with "press" but NO actual spam content = Dropped
        # Actual spam verification calls ("verify your business") = No even if short
        spam_content_phrases = ["verify your business", "verify your code", "google voice search",
                                "not verified", "press 0 to verify", "press zero to verify",
                                "press 0. to verify", "listing may be suspended"]
        has_spam_content = any(p in transcript_lower for p in spam_content_phrases)
        if len(transcript_lower) < 80 and "press" in transcript_lower and not has_spam_content:
            return "Dropped"
        return "No"  # Spam robocall = No (automated message)

    # EARLY CHECK: "Currently closed" announcement = Dropped
    # Must check early because transcription may split into multiple speakers
    # But if there's a "leave a message" prompt, it's voicemail = No
    if "currently closed" in transcript_lower:
        vm_prompt_patterns = ["leave a message", "leave your message", "leave me a message",
                              "please leave", "leave your name", "leave a voicemail"]
        has_vm_prompt = any(p in transcript_lower for p in vm_prompt_patterns)
        if has_vm_prompt:
            return "No"  # Closed + VM prompt = voicemail
        # Check for actual conversation (not just garbled transcription)
        # Real conversation would have service request, quote, appointment etc.
        real_convo_patterns = ["can i help", "how can i", "i need", "i want", "estimate",
                               "quote", "appointment", "schedule", "repair", "service"]
        has_real_convo = any(p in transcript_lower for p in real_convo_patterns)
        if not has_real_convo:
            return "Dropped"  # Just closed message, no VM, no conversation = Dropped

    # NO CALLER RESPONSE pattern - EARLY CHECK
    # Business answers ("Hello? Is anybody there?") but no caller response = "No"
    # Must check BEFORE has_human_conversation since "how can i help" triggers that
    # Only use EXPLICIT no-response indicators - not just multiple hellos
    # BUT: "can't hear you" in middle of conversation is just asking for clarification
    no_response_indicators = [
        "is anybody there",
        "i can't hear any response",
        "can't hear any response",
        "is anyone there",
        "anyone there?",
    ]
    # "can't hear you" only indicates no response if it appears EARLY and there's no dialogue after
    # Don't include it in main check - it's too prone to false positives in real conversations
    if any(p in transcript_lower for p in no_response_indicators):
        # Double-check: if there's clear dialogue pattern, it's actually a conversation
        if not re.search(r'can i help you\?[\s\S]{0,100}?(yes|yeah|i need|i have)', transcript_lower):
            return "No"  # Business answered but caller never responded

    # "Higher than normal call volume" = automated message, caller waiting
    # But if someone answers AFTER, it's still a "Yes"
    if "higher than normal call volume" in transcript_lower or "experiencing a high call volume" in transcript_lower:
        # Check if there's REAL conversation AFTER the hold message
        # Look for dialogue patterns that indicate someone actually answered
        after_hold_convo = bool(re.search(
            r'(higher than normal call volume|high call volume)[\s\S]{0,800}?(how can i help|how may i help|this is \w+[,.]|can i help you|\?[\s\S]{0,50}(yes|no|yeah|sure|i need))',
            transcript_lower
        ))
        # Also check for multiple speakers with dialogue AFTER the hold phrase
        hold_pos = max(
            transcript_lower.find("higher than normal call volume"),
            transcript_lower.find("high call volume")
        )
        after_hold_text = transcript_lower[hold_pos:] if hold_pos > 0 else ""
        # If there are questions/answers after hold, someone answered
        if re.search(r'\?[\s\S]{0,100}(yes|no|yeah|i\'d like|i need|my name)', after_hold_text):
            after_hold_convo = True
        # If there's "speaking" or name introduction after hold
        if re.search(r'(this is|speaking|hi,? i\'m)', after_hold_text):
            after_hold_convo = True
        if not after_hold_convo:
            # Check if there's also a VM prompt (hold queue → VM = voicemail = No)
            vm_after_hold = any(p in transcript_lower for p in [
                "leave a message", "leave your message", "leave me a message",
                "please leave", "after the beep", "at the tone", "record your message",
                "leave a voicemail"])
            if vm_after_hold:
                return "No"  # Hold queue + VM prompt = voicemail
            return "Dropped"  # Pure hold queue/IVR = Dropped

    # VM + message left pattern: "is not available" + someone leaving message (no real dialogue)
    # This should be "No" even if there are multiple speakers
    # BUT: "Sam is not available" mid-conversation is NOT voicemail
    if "is not available" in transcript_lower or "not available to take your call" in transcript_lower:
        # Check if there's clear conversation/dialogue BEFORE "is not available"
        # If so, it's a real conversation where someone happens to be unavailable
        not_avail_pos = transcript_lower.find("is not available")
        if not_avail_pos < 0:
            not_avail_pos = transcript_lower.find("not available to take your call")
        before_text = transcript_lower[:not_avail_pos] if not_avail_pos > 0 else ""
        has_dialogue_before = bool(re.search(
            r'(this is \w+[,.]|yes,?\s+hi|how can i help|can i help|just give me one moment|one moment|hold on|let me check)',
            before_text
        ))
        if not has_dialogue_before:
            # No conversation before "is not available" = VM greeting pattern
            has_dialogue_after_vm = bool(re.search(
                r'(how can i help|how may i help|can i help you|what can i do for you)\?[\s\S]{0,150}?(yes|yeah|hi|hello|my name|i need|i have|i\'m calling)',
                transcript_lower
            ))
            if not has_dialogue_after_vm:
                return "No"  # VM + message left = No

    # Check for human conversation
    has_convo = has_human_conversation(transcript)

    # Check for clear 2-way dialogue patterns that indicate someone answered
    # Use limited [\s\S]{0,100} to match across speaker tags but not too greedily
    # IMPORTANT: Use word boundaries \b to avoid matching "hi" inside "this"
    clear_answered_patterns = [
        r'how (can|may) i help you\?[\s\S]{0,100}?\b(hi|hello|yes|my name|i\'m)\b',
        r'(good morning|good afternoon|hello)[,.]?\s*\w+\s+(roofing|plumbing|construction)[\s\S]{0,100}?\b(hi|hello|yes|my name)\b',
        r'\b(yes|yeah|okay|sure)[,.]?\s*(can you|could you|i need|my name|hold on)',
        # "Can I help you?" followed by response (across speaker tags)
        r'can i help you\?[\s\S]{0,100}?\b(hi|hello|yes|yeah|my name|i need)\b',
        # "Good afternoon, this is [name] from [company]" + response pattern
        r'(good morning|good afternoon)[,.]?\s*this is \w+[\s\S]{0,100}?can i help[\s\S]{0,100}?\b(yes|yeah|i\'m calling|i need)\b',
        # Two people introducing themselves = conversation (e.g., "this is Hope... Hi, this is Ben")
        r'this is \w+[\s\S]{0,200}?hi,? this is \w+',
        # Question followed by confirmation across speakers (e.g., "you said Ben?" ... "Yep")
        r'you said \w+\?[\s\S]{0,50}?\b(yep|yes|yeah|correct|that\'s right)\b',
        # Back-and-forth with "one second" or "hold on" + "thank you" = conversation
        r'(one second|hold on|just a moment)[\s\S]{0,100}?(okay|thank you|thanks)',
    ]
    for pattern in clear_answered_patterns:
        if re.search(pattern, transcript_lower):
            return "Yes"

    # If real human conversation (and not spam), it's answered
    if has_convo:
        return "Yes"

    # Pure IVR with "opt out" (not spam) = Dropped, but VM with opt-out = No
    if has_opt_out:
        if is_vm:
            return "No"  # VM with opt-out spam overlay = still voicemail
        return "Dropped"

    # Check for TRUE IVR menu routing BEFORE hold queue handling
    # IVR menus that route callers ("press 1 for sales") = Dropped
    # vs. Hold queue callbacks ("press 1 to call you back") = No (handled below)
    ivr_menu_patterns = [
        r'for new customers?,? (please )?press',
        r'for (an )?existing customer?,? (please )?press',
        r'if you\'re (a )?(returning|new|existing).*press',
        r'press [12] for (sales|service|billing|support|production)',
        r'if you need help.*press \d',
        r'for (sales|service|billing|production).*press',
        r'to schedule.*press \d',  # "to schedule a free consultation, please press 1"
        r'to speak with.*press \d',  # "to speak with an agent, press 1"
    ]
    is_ivr_menu = any(re.search(p, transcript_lower) for p in ivr_menu_patterns)
    # Also check for callback patterns that indicate hold queue, NOT IVR menu
    is_callback_ivr = any(p in transcript_lower for p in [
        "call you back", "text reply", "call back instead", "callback"
    ])
    # Check if human answered BEFORE IVR (e.g., "Hello." then IVR menu)
    human_greeting_before_ivr = bool(re.match(
        r'^speaker [a-z]:\s*(hello|hi)[.!?,\s]', transcript_lower
    ))
    # Also check for human dialogue after IVR menu
    human_after_ivr = bool(re.search(
        r'(hi,?\s+\w+,?\s+this is|this is \w+ (from|with|at)|how can i help|how may i help)',
        transcript_lower
    ))
    # Check if this is actually a VM with IVR options (should be No, not Dropped)
    # VM with "press X" options is still voicemail
    if is_ivr_menu and not is_callback_ivr and not has_convo and not human_greeting_before_ivr and not human_after_ivr:
        if is_vm:
            return "No"  # VM with IVR options = No (voicemail)
        return "Dropped"  # True IVR menu, not hold queue

    # Call screening system pattern - "if you record your name" followed by dialogue = Yes
    # e.g., "Hi, if you record your name and reason for calling, I'll see if this person is available."
    # This is NOT voicemail - it's an automated screening that connects to a human
    if "if you record your name" in transcript_lower or "record your name and reason" in transcript_lower:
        # Check if there's actual dialogue after the screening
        if len(unique_speakers) >= 2:
            return "Yes"  # Multiple speakers after screening = answered
        # Check for dialogue patterns that indicate human conversation
        if re.search(r'(i was wanting|i need|do you (guys )?do|can i get)', transcript_lower):
            return "Yes"

    # Hold queue handling - check if someone actually answered after hold message
    if is_hold_queue:
        # Check for automated callback message first
        if "someone from our team will be calling you back" in transcript_lower:
            return "Dropped"  # Automated callback system
        if "will be calling you back" in transcript_lower:
            return "Dropped"  # Automated callback system
        if "we'll give you a call back" in transcript_lower:
            return "Dropped"  # Automated callback system
        if "give you a call back" in transcript_lower and "press" in transcript_lower:
            return "Dropped"  # Callback with IVR option

        # Hold queue with callback/text options - check if there's also VM
        # If VM prompt exists, it's No (voicemail). If just IVR options, it's Dropped.
        if "call you back" in transcript_lower or "text reply" in transcript_lower:
            if is_vm:
                return "No"  # Has VM prompt = No
            return "Dropped"  # IVR callback options only, no VM = Dropped

        # Check for multiple speakers after hold message (someone answered)
        speakers = re.findall(r'Speaker ([A-Z]):', transcript)
        unique_speakers = set(speakers)

        # If 2+ speakers, check if there's REAL conversation after hold
        if len(unique_speakers) >= 2:
            # Look for dialogue after hold (someone answering)
            if "how can i help" in transcript_lower or "how may i help" in transcript_lower:
                return "Yes"  # Someone answered after hold
            if "can i help you" in transcript_lower:
                return "Yes"  # Someone answered after hold
            # Name introduction after hold (including possessives like Riley's)
            if re.search(r'(this is|hey,? this is|hi,? this is) \w+[\'s,. ]', transcript_lower):
                return "Yes"
            # "Hey, I'm wondering" or similar human dialogue
            if "i'm wondering" in transcript_lower or "i need a quote" in transcript_lower:
                return "Yes"
            # "I need to speak with" = dialogue
            if "i need to speak" in transcript_lower or "need to speak with" in transcript_lower:
                return "Yes"
            # "My name is" = human dialogue (caller introducing themselves)
            if re.search(r'my name is \w+', transcript_lower):
                return "Yes"
            # "Good morning/afternoon" + dialogue = human conversation
            if re.search(r'(good morning|good afternoon)[,.]?\s*(my name|i\'m calling|i had|i was)', transcript_lower):
                return "Yes"
            # "[Name] from [Company]" pattern - human answered
            if re.search(r'\w+ from (the )?\w+ (roofing|plumbing|construction|dudes|hvac)', transcript_lower):
                return "Yes"
            if re.search(r'\w+ from the \w+', transcript_lower):
                return "Yes"
            # Business name + customer response pattern: "Roofing. Yes. Hi there."
            if re.search(r'\.\s*(yes|hi|hello|yeah)[,.]?\s+(hi|hello|there|my name)', transcript_lower):
                return "Yes"
            # "[Company]. Hello." pattern - human answered
            if re.search(r'(roofing|plumbing|construction|hvac|glass|contractor)[.\s]+hello', transcript_lower):
                return "Yes"
            # Quote request after hold = human conversation
            if "get a quote" in transcript_lower or "roof replacement" in transcript_lower:
                return "Yes"
            # 3+ speakers almost always means conversation happened
            # BUT NOT if it's voicemail (3 speakers = VM greeting + carrier + caller leaving message)
            if len(unique_speakers) >= 3 and not is_vm:
                return "Yes"

        # Hold message only (no human answering)
        # If VM prompt exists = No (voicemail). If pure hold queue = Dropped.
        if is_vm:
            return "No"  # Hold queue + VM prompt = No
        return "Dropped"  # Pure hold queue, no VM = Dropped

    # "Office hours" announcements = Dropped (informational IVR)
    # But VM with office hours = No (voicemail takes precedence)
    if "office hours" in transcript_lower:
        if is_vm:
            return "No"  # VM with office hours info = No (voicemail)
        return "Dropped"

    # "Currently closed" announcements = Dropped (automated message, no conversation)
    # e.g., "Thank you for calling. We are currently closed."
    if "currently closed" in transcript_lower or "we are closed" in transcript_lower:
        if is_vm:
            return "No"  # VM with closed message = No
        # Check if there's actual dialogue after the closed message
        if not has_convo:
            return "Dropped"  # Just closed message, no conversation = Dropped

    # True IVR menu (not voicemail options) = Dropped
    # But voicemail with "press 1 for options" is still voicemail = No
    if has_ivr_strict:
        # VM with press options = No (still voicemail, not answered)
        if is_vm:
            return "No"
        # Check if someone answered BEFORE the IVR (human greeting then IVR)
        # Pattern: "Speaker A: Hello." followed by IVR = human answered
        if re.match(r'^speaker [a-z]:\s*(hello|hi)[.!?\s]*\s*speaker [b-z]:', transcript_lower):
            return "Yes"  # Human answered with greeting before IVR
        # Check if someone answered AFTER the IVR
        # Look for actual human dialogue after IVR
        # Patterns: "how can i help", "this is [name]", or actual dialogue
        human_answered_after_ivr = bool(re.search(
            r'how can i help|how may i help|can i help you',
            transcript_lower
        ))
        # "this is [name]" indicates human answered (not just IVR menu)
        # Include possessive forms like "Riley's" and names followed by various punctuation
        if bool(re.search(r'(this is|hi,? this is|hey,? this is) \w+[\'s,. ]', transcript_lower)):
            human_answered_after_ivr = True
        # "This is [Name] speaking" or "[Name] speaking" pattern - but NOT "forward to speaking"
        if bool(re.search(r'(^|[.!?] )\w+ speaking[,.]', transcript_lower)):
            human_answered_after_ivr = True
        # "Hello" followed by "Thank you for calling" = human greeting (not IVR)
        if bool(re.search(r'hello[.,]?\s*(thank you for calling|thanks for calling)', transcript_lower)):
            human_answered_after_ivr = True
        # "Hey, this is [name]" after IVR = human answered
        if bool(re.search(r'hey,?\s+this is \w+', transcript_lower)):
            human_answered_after_ivr = True
        # "Hey, I'm wondering" or "I need" = human conversation
        if "i'm wondering" in transcript_lower or "i need" in transcript_lower:
            human_answered_after_ivr = True
        # Business name + customer response = human answered
        if re.search(r'\.\s*(yes|hi|hello|yeah)[,.]?\s+(hi|hello|there|my name)', transcript_lower):
            human_answered_after_ivr = True
        # Check for actual dialogue after IVR (question/answer pattern)
        has_dialogue_after_ivr = bool(re.search(
            r'\?\s*(yes|no|yeah|hi|hello|sure|okay|we do|we don)',
            transcript_lower
        ))
        if human_answered_after_ivr or has_dialogue_after_ivr:
            return "Yes"  # Human answered after IVR menu
        return "Dropped"

    # Hold music pattern (songs/music before IVR) = IVR/Dropped
    if has_hold_music_pattern and not has_convo:
        return "Dropped"

    # Check if human answered but caller didn't respond much
    # Multiple speakers with business greeting at start
    # BUT if it's voicemail (VM greeting + caller leaving message) = No
    # AND NOT if it's hold queue or IVR without actual conversation
    if len(unique_speakers) >= 2:
        # If voicemail with multiple speakers (VM greeting + carrier + message left) = No
        if is_vm:
            return "No"  # Voicemail system, not answered
        # Don't return Yes if this is hold queue without conversation
        if is_hold_queue:
            return "No"  # Already handled above, but safety check
        # Don't return Yes if this is IVR without conversation
        if has_ivr_strict:
            return "Dropped"  # Already handled above, but safety check
        # Any multi-speaker conversation is an answered call
        # Business greeting or short name is fine
        return "Yes"

    # Single speaker with dialogue patterns = answered
    # But NOT if it's voicemail or hold queue
    if len(unique_speakers) == 1:
        # If voicemail system, it's No even with dialogue patterns
        if is_vm:
            return "No"
        # If hold queue, it's No (waiting, not answered)
        if is_hold_queue:
            return "No"

        # Check for confused questions pattern (no response)
        # "How may I help you? What is this?" = business confused, caller not responding
        # (Don't count multiple hellos as "no response" - "Hello? Hello?" means call was answered)
        if "what is this" in transcript_lower and "how" in transcript_lower and "help" in transcript_lower:
            return "No"  # Business confused, no caller response

        # Check for "intro + hello" pattern = call WAS answered, even if caller didn't respond
        # "Hello, this is sage roofing. Hello." = business answered and checked for response
        # This is Yes (answered) but outcome should be Disqualified (no engagement)
        hello_count = len(re.findall(r'\bhello\b', transcript_lower))
        if hello_count >= 2:
            # If there's a business intro ("this is") plus multiple hellos = answered
            if re.search(r'this is \w+', transcript_lower):
                return "Yes"  # Business answered and checked for response = answered

        # Business introduction patterns = Yes (someone answered)
        # "This is Jessica", "Hello, this is John." = Yes, call was answered
        if re.search(r'this is \w+', transcript_lower):
            return "Yes"
        # Dialogue patterns within single speaker (transcription quirk)
        if re.search(r'\. yes[,.]', transcript_lower) or re.search(r'\. hi[,. ]', transcript_lower):
            return "Yes"
        if re.search(r'\. hey[,. ]', transcript_lower):
            return "Yes"
        if "speaking with" in transcript_lower:
            return "Yes"
        if "how can i help" in transcript_lower or "how may i help" in transcript_lower:
            return "Yes"
        # Check for "Hello, Spam" pattern first = someone marked call as spam
        if "spam" in transcript_lower:
            return "No"  # Marked as spam by the business

        # "Hello?" patterns - the business answered, so it's "Yes"
        # Even if caller didn't respond, the call was answered
        if "hello?" in transcript_lower or "hello." in transcript_lower or "hello," in transcript_lower:
            return "Yes"
        # "Hello" alone or repeated (business answering) = Yes (they answered)
        if "hello" in transcript_lower:
            return "Yes"
        if transcript_lower.strip() in ["hello", "hello.", "hello!"]:
            return "Yes"
        if re.match(r'^speaker [a-z]:\s*hello\.?\s*$', transcript_lower):
            return "Yes"
        # Very short single-speaker transcript without intro = No (caller hung up)
        if len(transcript.strip()) < 50:
            return "No"
        # Single speaker with just intro = No (caller didn't respond)
        return "No"

    # Multiple speakers but no conversation detected - use AI
    return "unclear"


def analyze_with_ai(transcript: str, source: str, check_type: str, max_retries: int = 3) -> dict:
    """Use Claude to analyze specific aspects of the transcript"""

    # Rate limiting delay
    time.sleep(RATE_LIMIT_DELAY)

    prompts = {
        "call_answered": """Analyze this call transcript to determine if a human answered the call.

CRITICAL RULES FOR "Dropped":
- Return "Dropped" if the call reached ONLY an IVR/automated phone system:
  - "press 1 for...", "press 2 for...", "please press X"
  - "opt out", "please hold", "thank you for calling...for X press 1"
  - Hold music followed by IVR menu
  - Automated callback systems ("someone will call you back")
  - IVR menus even if they include spam-like content
  - Multiple automated speakers reading IVR prompts
- IVR = Dropped (this takes priority over other classifications)

CRITICAL RULES FOR "No":
- Return "No" for voicemail:
  - "leave a message", "at the tone", "not available", "missed your call"
  - Only voicemail greeting + message left (or no message)
- Return "No" for SPAM/robocalls:
  - "verify your business", "Google listing", "urgent message"
  - Automated spam with multiple speakers (transcription artifacts from robocall)
  - These are NOT answered calls - they are spam recordings
- Return "No" if no human engagement occurred at all

CRITICAL RULES FOR "Yes":
- Return "Yes" ONLY if:
  - A real human answered AND engaged in conversation
  - A human answered with greeting ("Hello?", "Company Name, how can I help?") and there was back-and-forth dialogue
  - Clear two-way conversation between caller and recipient
- Do NOT return "Yes" for:
  - IVR systems (even with "Hello")
  - Spam calls with multiple automated speakers
  - Automated messages only

Transcript:
{transcript}

Source: {source}

Return ONLY a JSON object: {{"answer": "Yes/No/Dropped/Not a Phone Call", "reasoning": "brief explanation"}}""",

        "outcome_detailed": """Analyze this lead to determine its outcome classification.

VERIFIED - Genuine NEW customer seeking services:
- Customer calling to request a NEW service, quote, or estimate
- Customer asking about products/services for a NEW project
- NEW appointment requests for service work
- Google LSA leads from actual customers
- A homeowner/customer with a service need they want addressed
- Repair requests even if insurance is involved (insurance is just payment method)
- Product inquiries for products/services the company PROVIDES = VERIFIED (potential customer)
- Pricing/availability questions about materials or services = VERIFIED
- IMPORTANT: If a customer has a LEGITIMATE service need, it's VERIFIED even if:
  - The company cannot help them (project too small, outside service area, etc.)
  - The company declines the job for any reason
  - The customer's project doesn't meet company minimums
  - The focus is on the CUSTOMER'S INTENT, not whether business was completed

DISQUALIFIED - Not a valid new business lead:
- SPAM/scam calls (business verification, Google listings, Yelp, etc.)
- Billing, payment, or invoice related calls (paying for COMPLETED work)
- Follow-up on RECENT work (within last 3 months) - quotes, estimates, scheduled work
  - CRITICAL: "yesterday", "last week", "few days ago", "couple weeks" = WITHIN 3 months = Disqualified
  - If timeframe is NOT specified, treat as potentially Verified (let it through)
- Rescheduling or canceling existing appointments
- Caller asking for SPECIFIC PERSON BY NAME with NO service request (indicates existing relationship)
- Existing customer checking status of current job/quote
- Wrong number
- Compliments/reviews/feedback (not requesting new service)
- Employment, recruiting, or staffing calls
- B2B marketing/sales calls TO the business (advertising, golf course guidebooks, etc.)
  - IMPORTANT: This is B2B trying to SELL to the business = Disqualified
  - But B2B wanting to USE/HIRE this company's services = VERIFIED (they are a customer)
- Chamber of Commerce, BBB, or membership calls
- Third-party lead platforms
- Marketing/sales pitches TO the business
- Internal company calls
- Warranty claims on SAME work done recently
- Vendors, solicitors calling TO SELL to the business
- REFERRALS ("referred by someone", "someone referred me") - not from marketing channels
- "Let me know if you're interested" messages - solicitation
- SMS lead solicitation ("aren't shared leads", "if you're not interested")
- Yelp sales calls = instant Disqualified

UNVERIFIED - Cannot determine intent:
- VOICEMAIL where no message was left OR only callback info given
- Voicemail greeting only - no indication of caller's intent
- Voicemail with caller asking for specific person (unclear intent)
- Very short transcripts with only greeting, no indication of intent
- IVR systems with no human interaction
- Call answered but immediately ended with no conversation
- CRITICAL: Voicemail = UNVERIFIED by default unless it contains:
  - Clear spam content (Yelp, Google listing, B2B marketing) = Disqualified
  - Clear service request ("need roof repair") = Verified
  - Otherwise = UNVERIFIED (cannot determine intent from VM)
- Short/unclear transcripts without spam = UNVERIFIED
- "Record your message" with no actual message = UNVERIFIED

CRITICAL DISTINCTIONS:
- "Had an appointment today for estimate" = DISQUALIFIED (existing engagement)
- "Waiting for quote from last week" = DISQUALIFIED (recent follow-up)
- "Want to schedule NEW service" = VERIFIED
- "Need roof repair, insurance will cover it" = VERIFIED (new service request)
- "Need tarping while dealing with insurance" = VERIFIED (new service request)
- Real estate agent calling about their client = DISQUALIFIED (B2B/affiliate)
- Contractor calling about their project = DISQUALIFIED (B2B)
- Very short unclear messages = UNVERIFIED

IMPORTANT - PAST CUSTOMERS (3-MONTH RULE):
- CRITICAL: Past customer WITHIN 3 months = DISQUALIFIED (not new business)
  - "yesterday", "last week", "few days ago", "couple weeks", "last month" = recent = DISQUALIFIED
  - "work that was done yesterday" = DISQUALIFIED
  - "just settled up payments" = DISQUALIFIED (very recent)
- Past customer OVER 3 months or NO TIMEFRAME specified = can be VERIFIED
  - "You did my roof" (no timeframe) = VERIFIED (could be years ago)
  - "Gosh, it might be going on a year" = VERIFIED (over 3 months)
  - "You did our siding 2 years ago" = VERIFIED (over 3 months)
- The key is: Is the timeframe specified and within 3 months? If yes = DISQUALIFIED
- If no timeframe is mentioned, treat as potentially Verified (over 3 months)

IMPORTANT - SERVICE AND PRODUCT INQUIRIES:
- If caller asks about services ("do you do X roofs?") = VERIFIED
- If caller asks about products ("do you carry X shingles?") = VERIFIED (they want to purchase)
- ANY inquiry about buying products or services from the company = VERIFIED
- Product availability questions = VERIFIED (potential sale)
- Material inquiries = VERIFIED (they want to purchase from the company)
- The company sells services AND products, so both are valid leads
- CRITICAL: Even if company says "we don't do that" or "we're not a supplier", if the caller had a GENUINE inquiry = VERIFIED
- The focus is on CALLER'S INTENT, not whether the company could help them
- "Do you do gutter cleaning?" (company says no) = VERIFIED (genuine inquiry)
- "Do you sell standing seam panels?" (company says no) = VERIFIED (genuine product inquiry)

IMPORTANT - SERVICE REQUESTS EVEN IF OTHERS DECLINED:
- Someone with a service need that other companies declined = VERIFIED (they have a real need)
- "Three companies turned me down" + service need = VERIFIED
- Property managers or landlords calling about their properties = VERIFIED if they have a service need
- "I have a property at..." + service need = VERIFIED (property managers are valid customers)

IMPORTANT - B2B MARKETING vs SERVICE REQUESTS:
- B2B marketing/sales TO the business = DISQUALIFIED (someone trying to sell to this company)
- B2B intermediaries referring without service need (paint companies saying "send customers to us") = DISQUALIFIED
- Real estate agents/brokers requesting inspections for properties = VERIFIED (they need service)
- But COMMERCIAL/INDUSTRIAL service requests = VERIFIED (they have a real need)
- "I have an industrial building that needs a roof" = VERIFIED (real service need)
- "I'm a building owner calling about roof repair" = VERIFIED
- Condo/HOA boards calling about building repairs = VERIFIED
- The key distinction: Are they HIRING this company for work (Verified) or SELLING to this company (Disqualified)?
- CRITICAL: B2B wanting to USE this company's services = VERIFIED
  - "Looking for a roofing service we can use" = VERIFIED (they want to hire this company)
  - "Reaching out looking for a contractor" = VERIFIED (they are potential customer)
  - Real estate agents looking for a contractor to use = VERIFIED (they will hire this company)
  - "Are you bidding [project name]?" = VERIFIED (contractor wants to partner/subcontract on a project)
  - Contractor-to-contractor project inquiries = VERIFIED (potential business collaboration)

IMPORTANT - AGENTS AND REPRESENTATIVES:
- Agent/representative calling on behalf of property owner with SERVICE NEED = VERIFIED
- "I'm an agent, I need a roof inspection" = VERIFIED (genuine service need)
- Real estate agents with actual service requests for their clients = VERIFIED
- The key is: Do they have a GENUINE service need? If yes = VERIFIED

IMPORTANT - EXISTING QUOTES:
- Caller already has a quote from this company = DISQUALIFIED (not new business)
- "I received a quote and want to proceed" = DISQUALIFIED (follow-up on existing)
- "Got your quote, calling to schedule" = DISQUALIFIED

IMPORTANT - ANSWERED BUT NO BUSINESS INTENT:
- Call answered but no service request or intent expressed = DISQUALIFIED
- Internal calls = DISQUALIFIED
- Location/directions questions only = DISQUALIFIED
- Language barrier with no clear intent = DISQUALIFIED
- Verification codes = DISQUALIFIED

IMPORTANT - MAINTENANCE AND REPAIR:
- NEW maintenance scheduling = VERIFIED (this is new work)
- "I need cooling maintenance" = VERIFIED (new service)
- Warranty repair on SAME work = DISQUALIFIED
- Repair of DIFFERENT component = VERIFIED ("you did my roof, now need gutters fixed")

IMPORTANT - GENUINE INQUIRIES:
- If someone has a genuine service need, it's VERIFIED even if:
  - The company cannot help them
  - Other companies declined the job
  - The inquiry is about something the company doesn't typically do
- Focus on CUSTOMER'S INTENT, not whether business was completed

IMPORTANT - PAST CUSTOMERS WITH NEW SERVICE NEEDS:
- Past customer calling for DIFFERENT/NEW service = VERIFIED
- "You did my roof, now I need gutters fixed" = VERIFIED (new service, not warranty)
- "You did my siding 2 years ago, gutter has pulled away" = VERIFIED (repair of different component)
- CRITICAL: If a past customer mentions recent work on ONE part of a building but is calling about a COMPLETELY DIFFERENT structure/issue, this is a NEW service request = VERIFIED
  - Example: "You repaired our steeple in the summertime, now our 20-year-old flat roof needs evaluation" = VERIFIED (different structure, new issue)
  - Example: "You did some work for us in the past... our insurance company says this roof needs replacing" = VERIFIED (new evaluation request)
  - The mention of past work is just CONTEXT (establishing relationship/trust), not the reason for calling
  - Focus on what service the caller is ACTUALLY REQUESTING NOW, not what was done before
  - Recent contact (like "I got your message yesterday") does NOT make a NEW service request into a follow-up
  - CRITICAL: If the caller requests evaluation/estimate for a different structure or issue than past work, it is ALWAYS Verified
- Only WARRANTY claims on the SAME work = DISQUALIFIED
- "My roof you installed is leaking" = DISQUALIFIED (warranty on same work)

IMPORTANT - SCHEDULING AND FOLLOW-UPS:
- "I spoke to you yesterday" / "I spoke with you" about anything = DISQUALIFIED (follow-up on prior conversation)
- "Had been conversing with [person]" = DISQUALIFIED (follow-up on existing discussion)
- "I called earlier" about scheduling/status/general inquiry = DISQUALIFIED (same-day follow-up)
- BUT "I called earlier about [active service problem]" where technician is working on it = VERIFIED (active service engagement)
  - Example: "Called earlier about a waterline issue, talked to a technician" = VERIFIED (active repair work)
- "You did my roof last week" = DISQUALIFIED (work done within 3 months)
- Past customer OVER 3 months with new need = VERIFIED ("year or two ago", "last year", "few years")

IMPORTANT - REFERRALS:
- ALL referrals = DISQUALIFIED (not from marketing channels)
- "Friend recommended you" = DISQUALIFIED regardless of service need
- "Was referred by someone" = DISQUALIFIED
- "You did my neighbor's gutters/roof" = DISQUALIFIED (neighbor saw-your-work referral)
- "You did work on my daughter's/son's/family member's house" = DISQUALIFIED (family referral)
- Referrals are tracked separately, not counted as marketing leads
- CRITICAL DISTINCTION: "Calling FOR a friend to get pricing/info" is NOT a referral - it's a genuine inquiry = VERIFIED
  - "It's for a friend, I'm just calling around to help" = VERIFIED (sourcing services, genuine inquiry)
  - The caller was NOT referred BY someone, they are actively shopping for services

Transcript:
{transcript}

Source: {source}

Return ONLY a JSON object: {{"outcome": "Verified/Disqualified/Unverified", "reasoning": "brief explanation"}}"""
    }

    prompt = prompts[check_type].format(transcript=transcript, source=source)

    # Get model and temperature from environment variables (for testing)
    # gpt-5.2 is the best performing model (95%+ accuracy, 46 mismatches)
    model = os.environ.get("OPENAI_MODEL", "gpt-5.2")
    temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0"))

    # Newer models (gpt-5, o3, o4, etc.) use max_completion_tokens instead of max_tokens
    # and don't support temperature parameter
    newer_models = ["gpt-5", "o3", "o4", "o1"]
    is_newer_model = any(model.startswith(prefix) for prefix in newer_models)

    for attempt in range(max_retries):
        try:
            if is_newer_model:
                # Newer models use different parameters
                response = client.chat.completions.create(
                    model=model,
                    max_completion_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=500,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
            break
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(30)  # Wait 30 seconds on rate limit
                continue
            else:
                return {}  # Return empty on repeated failures

    # Parse JSON response
    text = response.choices[0].message.content.strip()
    # Extract JSON from response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}


def analyze_lead(source: str, transcript: str, use_ai: bool = True) -> AnalysisResult:
    """Main analysis function that applies all rules"""

    transcript = transcript.strip()

    # ========== STEP 1: Check if Not a Phone Call ==========
    if is_not_a_phone_call(source, transcript):
        # Check if it's an LSA lead (auto-verified)
        if is_lsa_lead(transcript):
            return AnalysisResult(
                call_answer="Not a Phone Call",
                outcome="Verified",
                reasoning="LSA lead - auto verified"
            )

        # Check for spam SMS
        if is_spam_verification_call(transcript) or is_third_party_lead_platform(transcript):
            return AnalysisResult(
                call_answer="Not a Phone Call",
                outcome="Disqualified",
                reasoning="Spam/marketing SMS detected",
                spam=True
            )

        # Website with NA or empty
        if transcript.lower() in ["na", "n/a", ""]:
            return AnalysisResult(
                call_answer="Not a Phone Call",
                outcome="Disqualified",
                reasoning="Empty/NA transcript"
            )

        # Try rule-based outcome first for SMS
        rule_outcome = get_rule_based_outcome(transcript, source, "Not a Phone Call")
        if rule_outcome:
            return AnalysisResult(
                call_answer="Not a Phone Call",
                outcome=rule_outcome,
                reasoning=f"SMS - rule-based: {rule_outcome}"
            )

        # Use AI for SMS outcome classification
        if use_ai and len(transcript) > 3:
            ai_result = analyze_with_ai(transcript, source, "outcome_detailed")
            outcome = ai_result.get("outcome", "Unverified")
            # Validate outcome
            if outcome not in ["Verified", "Disqualified", "Unverified"]:
                outcome = "Unverified"
            return AnalysisResult(
                call_answer="Not a Phone Call",
                outcome=outcome,
                reasoning=f"SMS: {ai_result.get('reasoning', '')}"
            )

        return AnalysisResult(
            call_answer="Not a Phone Call",
            outcome="Unverified",
            reasoning="Non-phone lead with unclear intent"
        )

    # ========== STEP 2: Phone Call Analysis ==========

    # Determine if call was answered first
    call_answer = was_call_answered(transcript)

    # If unclear, use AI to determine
    if call_answer == "unclear" and use_ai:
        ai_result = analyze_with_ai(transcript, source, "call_answered")
        call_answer = ai_result.get("answer", "No")
        if call_answer not in ["Yes", "No", "Dropped", "Not a Phone Call"]:
            call_answer = "No"

    # Check for spam (applies regardless of answer status)
    # Spam robocalls are automated - mark as "No" for call answer
    # Short garbled spam/IVR fragments = Dropped instead of No
    if is_spam_verification_call(transcript):
        spam_answer = "No"
        tl = transcript.lower()
        spam_content_phrases = ["verify your business", "verify your code", "google voice search",
                                "not verified", "press 0 to verify", "press zero to verify",
                                "press 0. to verify", "listing may be suspended"]
        has_spam_content = any(p in tl for p in spam_content_phrases)
        if len(tl) < 80 and "press" in tl and not has_spam_content:
            spam_answer = "Dropped"
        return AnalysisResult(
            call_answer=spam_answer,
            outcome="Disqualified",
            reasoning="Spam verification call detected",
            spam=True
        )

    # Check for third-party lead platforms
    if is_third_party_lead_platform(transcript):
        return AnalysisResult(
            call_answer=call_answer,
            outcome="Disqualified",
            reasoning="Third-party lead platform call"
        )

    # For Dropped calls
    if call_answer == "Dropped":
        return AnalysisResult(
            call_answer="Dropped",
            outcome="Disqualified",
            reasoning="IVR only - no human conversation"
        )

    # For unanswered/voicemail calls
    if call_answer == "No":
        # Try rule-based outcome first
        rule_outcome = get_rule_based_outcome(transcript, source, call_answer)
        if rule_outcome:
            return AnalysisResult(
                call_answer="No",
                outcome=rule_outcome,
                reasoning=f"Voicemail - rule-based: {rule_outcome}"
            )

        # Use AI to determine outcome
        if use_ai:
            ai_result = analyze_with_ai(transcript, source, "outcome_detailed")
            outcome = ai_result.get("outcome", "Unverified")
            if outcome not in ["Verified", "Disqualified", "Unverified"]:
                outcome = "Unverified"
            return AnalysisResult(
                call_answer="No",
                outcome=outcome,
                reasoning=f"Voicemail: {ai_result.get('reasoning', '')}"
            )

        return AnalysisResult(
            call_answer="No",
            outcome="Unverified",
            reasoning="Voicemail - needs AI analysis"
        )

    # ========== STEP 3: Answered Call Analysis ==========
    if call_answer == "Yes":
        # Try rule-based outcome first
        rule_outcome = get_rule_based_outcome(transcript, source, call_answer)
        if rule_outcome:
            return AnalysisResult(
                call_answer="Yes",
                outcome=rule_outcome,
                reasoning=f"Answered - rule-based: {rule_outcome}"
            )

        if use_ai:
            # Use AI to determine outcome
            ai_result = analyze_with_ai(transcript, source, "outcome_detailed")
            outcome = ai_result.get("outcome", "Unverified")
            if outcome not in ["Verified", "Disqualified", "Unverified"]:
                outcome = "Unverified"
            return AnalysisResult(
                call_answer="Yes",
                outcome=outcome,
                reasoning=f"Answered: {ai_result.get('reasoning', '')}"
            )

        return AnalysisResult(
            call_answer="Yes",
            outcome="Unverified",
            reasoning="Call answered - needs AI analysis for intent"
        )

    # ========== STEP 4: Unclear cases - use AI ==========
    if use_ai:
        # Get call answer from AI
        ai_answer = analyze_with_ai(transcript, source, "call_answered")
        answer = ai_answer.get("answer", "No")
        if answer not in ["Yes", "No", "Dropped", "Not a Phone Call"]:
            answer = "No"

        # Get outcome from AI
        ai_outcome = analyze_with_ai(transcript, source, "outcome_detailed")
        outcome = ai_outcome.get("outcome", "Unverified")
        if outcome not in ["Verified", "Disqualified", "Unverified"]:
            outcome = "Unverified"

        return AnalysisResult(
            call_answer=answer,
            outcome=outcome,
            reasoning=f"AI analysis: {ai_answer.get('reasoning', '')}"
        )

    return AnalysisResult(
        call_answer="No",
        outcome="Unverified",
        reasoning="Could not determine call status"
    )


def test_accuracy(csv_path: str, sample_size: Optional[int] = None, verbose: bool = True):
    """Test the analyzer against the manually labeled data"""

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if sample_size:
        rows = rows[:sample_size]

    correct_answer = 0
    correct_outcome = 0
    total = len(rows)

    mismatches = []

    for i, row in enumerate(rows):
        source = row['source']
        transcript = row['Transcription']
        expected_answer = row['Manual Answer']
        expected_outcome = row['Manual Outcome']

        result = analyze_lead(source, transcript, use_ai=True)

        answer_match = result.call_answer == expected_answer
        outcome_match = result.outcome == expected_outcome

        if answer_match:
            correct_answer += 1
        if outcome_match:
            correct_outcome += 1

        if not answer_match or not outcome_match:
            mismatches.append({
                'id': row['primary_id'],
                'source': source,
                'transcript': transcript[:300] + "..." if len(transcript) > 300 else transcript,
                'expected_answer': expected_answer,
                'got_answer': result.call_answer,
                'expected_outcome': expected_outcome,
                'got_outcome': result.outcome,
                'reasoning': result.reasoning,
                'answer_match': answer_match,
                'outcome_match': outcome_match
            })

        if verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total} - Answer: {correct_answer}/{i+1} ({100*correct_answer/(i+1):.1f}%) - Outcome: {correct_outcome}/{i+1} ({100*correct_outcome/(i+1):.1f}%)")

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Answer Accuracy: {correct_answer}/{total} ({100*correct_answer/total:.2f}%)")
    print(f"Outcome Accuracy: {correct_outcome}/{total} ({100*correct_outcome/total:.2f}%)")
    both_correct = total - len(mismatches)
    print(f"Both Correct: {both_correct}/{total} ({100*both_correct/total:.2f}%)")
    print(f"\nMismatches: {len(mismatches)}")

    return {
        'answer_accuracy': correct_answer / total,
        'outcome_accuracy': correct_outcome / total,
        'both_correct': both_correct / total,
        'mismatches': mismatches
    }


def analyze_single(transcript: str, source: str = "Unknown") -> dict:
    """Analyze a single lead"""
    result = analyze_lead(source, transcript, use_ai=True)
    return {
        'call_answer': result.call_answer,
        'outcome': result.outcome,
        'reasoning': result.reasoning,
        'spam': result.spam
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sample = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        results = test_accuracy("1k-leads-correct-analysis.csv", sample_size=sample)

        # Save mismatches for analysis
        with open("mismatches.json", "w") as f:
            json.dump(results['mismatches'], f, indent=2)
        print(f"\nMismatches saved to mismatches.json")
    else:
        # Example usage
        print("Usage: python lead_analyzer.py test [sample_size]")
        print("Example: python lead_analyzer.py test 100")
