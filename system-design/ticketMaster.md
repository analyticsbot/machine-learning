# Ticketmaster System Design

This document summarises key themes and insights from the video "System Design Interview: Design Ticketmaster w/ a Ex-Meta Staff Engineer". It highlights the process and considerations for designing a ticket booking system like Ticketmaster.

## 1. Requirements Gathering:

### Functional Requirements:
- Users should be able to book tickets.
- Users should be able to view event details.
- Users should be able to search for events.

### Non-Functional Requirements:
- **Consistency:** Prioritise strong consistency for booking tickets to prevent double bookings. "We need to make sure that no ticket is assigned to more than one user."
- **Availability:** Aim for high availability for searching and viewing events.
- **Scalability:** System must handle surges in traffic from popular events.
- **Read/Write Ratio:** System will have significantly more reads than writes (estimated 100:1).

## 2. Core Entities and API Design:

### Entities:
- **Event** (with attributes like name, description, venue ID, performer ID, tickets)
- **Venue** (with location, seat map)
- **Performer**
- **Ticket** (with seat, price, status)

### APIs:
- `GET /event/{eventID}` to view event details.
- `GET /search` to search for events based on criteria like term, location, type, date.

### Two-phase booking process:
- `POST /booking/reserve` to reserve a ticket (ticketID as input).
- `PUT /booking/confirm` to confirm purchase (ticketID, payment details).

## 3. High-Level Design:

- Client interacts with the system through an API Gateway.
- Microservice architecture with dedicated services for event management, search, and booking.
- **Database:** PostgreSQL chosen for strong consistency and relational capabilities.
- Simple search implemented with SQL queries (acknowledging the need for optimisation).
- Ticket reservation implemented using a distributed lock in Redis with TTL to handle expirations gracefully.

## 4. Deep Dives & Optimisations:

### Low Latency Search:
- Implement **Elasticsearch** for optimised search using inverted indexes and geospatial capabilities.
- Utilise **change data capture (CDC)** to synchronise data between PostgreSQL and Elasticsearch.
- Consider **caching mechanisms** like node query caching, Redis, or CDN for further performance improvements.

### Scalability for Popular Events:
- Implement a virtual waiting queue using Redis to manage high traffic and improve user experience during surges.
- Consider real-time seat map updates using long polling or **Server-Sent Events (SSE)** to provide accurate availability information to users.

### Horizontal Scaling:
- Leverage load balancers and auto-scaling capabilities provided by cloud platforms like AWS.

### Read Optimisations:
- Cache event, venue, and performer data in Redis to reduce load on the primary database.

## 5. Conclusion:

The final design aims to address all functional and non-functional requirements, balancing consistency, availability, scalability, and performance. It highlights the importance of iterative design and the need to consider trade-offs and potential bottlenecks during peak usage.


### Video from - https://www.youtube.com/watch?v=fhdPyoO6aXI&list=PL5q3E8eRUieWtYLmRU3z94-vGRcwKr9tM
