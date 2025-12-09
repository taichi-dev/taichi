# RFC: The RFC (Request for Comments) Process

* Author(s): [Ye Kuang](https://github.com/k-ye)
* Date: 2022-04-10
* Relevant Issue: N/A

---

# TL;DR

This doc proposes the standard way to make design decisions in Taichi.  The “RFC” (request for comments) process is intended to provide a consistent and controlled path for new features to land in Taichi, so that all stakeholders can be confident about the evolving direction.

# Background

Throughout the development of Taichi, we often hear new developers complaining about the steep learning curve of Taichi's codebase. One reason is that, currently the design decisions are spread across different places and hard to track.

RFC process is a well-established mechanism in the open source communities to solve this problem. It provides a common place for developers to discuss design decisions publicly. It also serves as a doc library for people who want to learn about Taichi's architecture, evolving history, etc.

# Goals

* Provides a way for Taichi developers to discuss and to reach a consensus on any major changes openly and efficiently.
* Serves as Taichi's public technical documentations.

# Detailed Design

1. Before initiating an RFC, it is recommended to file a Github Issue first to briefly describe the idea.
2. Make a copy of [`yyyymmdd-rfc-template.md`](yyyymmdd-rfc-template.md), fill in all the required sections.
3. Send a new PR with this RFC file, then go through the normal PR process.
4. If the relevant community members decide to approve this RFC:
   1. In case you haven't done step 1, now is a good time to open a new issue for tracking the works proposed by this RFC.
   2. Fill in the issue URL in `Relevant Issue`.
   3. Merge this PR & enjoy coding.
5. Otherwise if the members decide not to approve it, either re-iterate on the RFC and re-request the review, or simply drop it.

# Alternatives

## Other forms of documentation

RFC is not the only documentation system. We do have other forms of documentation, including [docsite](https://docs.taichi-lang.org/), technical blogs and design docs. Unlike RFCs, design docs will be written in a retrospective way to cover existing system's design and implementation.

## RFC ID

Instead of using date, there are other options for the RFC ID:

1. Use an incremental number as the RFC ID. For example, [Rust](https://github.com/rust-lang/rfcs/blob/master/0000-template.md) has a four-digit ID system. However, we are very ambitious developers and feel like it is possible to overflow any given number of digits.
2. Use the PR number associated with this RFC. This will work, but can make the workflow a bit cumbersome.

Date can conflict if multiple RFCs are proposed on the same day. However, given that 1) date is not the only identifier of the RFC (title is also included), and 2) TensorFlow [is using date](https://github.com/tensorflow/community/tree/master/rfcs), this should be fine.

# FAQ

1. When should I initiate a new RFC?

   There isn't a standard answer here. Rust's RFC guideline summarizes the situation pretty well:

   > You need to follow this process if you intend to make "substantial" changes to Rust, Cargo, Crates.io, or the RFC process itself. What constitutes a "substantial" change is evolving based on community norms and varies depending on what part of the ecosystem you are proposing to change...
