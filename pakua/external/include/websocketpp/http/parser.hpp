/*
 * Copyright (c) 2014, Peter Thorson. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the WebSocket++ Project nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL PETER THORSON BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef HTTP_PARSER_HPP
#define HTTP_PARSER_HPP

#include <algorithm>
#include <map>
#include <string>
#include <utility>

#include <websocketpp/utilities.hpp>
#include <websocketpp/http/constants.hpp>

namespace websocketpp {
namespace http {
namespace parser {

namespace state {
    enum value {
        method,
        resource,
        version,
        headers
    };
}

namespace body_encoding {
    enum value {
        unknown,
        plain,
        chunked
    };
}

typedef std::map<std::string, std::string, utility::ci_less > header_list;

/// Read and return the next token in the stream
/**
 * Read until a non-token character is found and then return the token and
 * iterator to the next character to read
 *
 * @param begin An iterator to the beginning of the sequence
 * @param end An iterator to the end of the sequence
 * @return A pair containing the token and an iterator to the next character in
 * the stream
 */
template <typename InputIterator>
std::pair<std::string,InputIterator> extract_token(InputIterator begin,
    InputIterator end)
{
    InputIterator it = std::find_if(begin,end,&is_not_token_char);
    return std::make_pair(std::string(begin,it),it);
}

/// Read and return the next quoted string in the stream
/**
 * Read a double quoted string starting at `begin`. The quotes themselves are
 * stripped. The quoted value is returned along with an iterator to the next
 * character to read
 *
 * @param begin An iterator to the beginning of the sequence
 * @param end An iterator to the end of the sequence
 * @return A pair containing the string read and an iterator to the next
 * character in the stream
 */
template <typename InputIterator>
std::pair<std::string,InputIterator> extract_quoted_string(InputIterator begin,
    InputIterator end)
{
    std::string s;

    if (end == begin) {
        return std::make_pair(s,begin);
    }

    if (*begin != '"') {
        return std::make_pair(s,begin);
    }

    InputIterator cursor = begin+1;
    InputIterator marker = cursor;

    cursor = std::find(cursor,end,'"');

    while (cursor != end) {
        // either this is the end or a quoted string
        if (*(cursor-1) == '\\') {
            s.append(marker,cursor-1);
            s.append(1,'"');
            ++cursor;
            marker = cursor;
        } else {
            s.append(marker,cursor);
            ++cursor;
            return std::make_pair(s,cursor);
        }

        cursor = std::find(cursor,end,'"');
    }

    return std::make_pair("",begin);
}

/// Read and discard one unit of linear whitespace
/**
 * Read one unit of linear white space and return the iterator to the character
 * afterwards. If `begin` is returned, no whitespace was extracted.
 *
 * @param begin An iterator to the beginning of the sequence
 * @param end An iterator to the end of the sequence
 * @return An iterator to the character after the linear whitespace read
 */
template <typename InputIterator>
InputIterator extract_lws(InputIterator begin, InputIterator end) {
    InputIterator it = begin;

    // strip leading CRLF
    if (end-begin > 2 && *begin == '\r' && *(begin+1) == '\n' &&
        is_whitespace_char(static_cast<unsigned char>(*(begin+2))))
    {
        it+=3;
    }

    it = std::find_if(it,end,&is_not_whitespace_char);
    return it;
}

/// Read and discard linear whitespace
/**
 * Read linear white space until a non-lws character is read and return an
 * iterator to that character. If `begin` is returned, no whitespace was
 * extracted.
 *
 * @param begin An iterator to the beginning of the sequence
 * @param end An iterator to the end of the sequence
 * @return An iterator to the character after the linear whitespace read
 */
template <typename InputIterator>
InputIterator extract_all_lws(InputIterator begin, InputIterator end) {
    InputIterator old_it;
    InputIterator new_it = begin;

    do {
        // Pull value from previous iteration
        old_it = new_it;

        // look ahead another pass
        new_it = extract_lws(old_it,end);
    } while (new_it != end && old_it != new_it);

    return new_it;
}

/// Extract HTTP attributes
/**
 * An http attributes list is a semicolon delimited list of key value pairs in
 * the format: *( ";" attribute "=" value ) where attribute is a token and value
 * is a token or quoted string.
 *
 * Attributes extracted are appended to the supplied attributes list
 * `attributes`.
 *
 * @param [in] begin An iterator to the beginning of the sequence
 * @param [in] end An iterator to the end of the sequence
 * @param [out] attributes A reference to the attributes list to append
 * attribute/value pairs extracted to
 * @return An iterator to the character after the last atribute read
 */
template <typename InputIterator>
InputIterator extract_attributes(InputIterator begin, InputIterator end,
    attribute_list & attributes)
{
    InputIterator cursor;
    bool first = true;

    if (begin == end) {
        return begin;
    }

    cursor = begin;
    std::pair<std::string,InputIterator> ret;

    while (cursor != end) {
        std::string name;

        cursor = http::parser::extract_all_lws(cursor,end);
        if (cursor == end) {
            break;
        }

        if (first) {
            // ignore this check for the very first pass
            first = false;
        } else {
            if (*cursor == ';') {
                // advance past the ';'
                ++cursor;
            } else {
                // non-semicolon in this position indicates end end of the
                // attribute list, break and return.
                break;
            }
        }

        cursor = http::parser::extract_all_lws(cursor,end);
        ret = http::parser::extract_token(cursor,end);

        if (ret.first.empty()) {
            // error: expected a token
            return begin;
        } else {
            name = ret.first;
            cursor = ret.second;
        }

        cursor = http::parser::extract_all_lws(cursor,end);
        if (cursor == end || *cursor != '=') {
            // if there is an equals sign, read the attribute value. Otherwise
            // record a blank value and continue
            attributes[name].clear();
            continue;
        }

        // advance past the '='
        ++cursor;

        cursor = http::parser::extract_all_lws(cursor,end);
        if (cursor == end) {
            // error: expected a token or quoted string
            return begin;
        }

        ret = http::parser::extract_quoted_string(cursor,end);
        if (ret.second != cursor) {
            attributes[name] = ret.first;
            cursor = ret.second;
            continue;
        }

        ret = http::parser::extract_token(cursor,end);
        if (ret.first.empty()) {
            // error : expected token or quoted string
            return begin;
        } else {
            attributes[name] = ret.first;
            cursor = ret.second;
        }
    }

    return cursor;
}

/// Extract HTTP parameters
/**
 * An http parameters list is a comma delimited list of tokens followed by
 * optional semicolon delimited attributes lists.
 *
 * Parameters extracted are appended to the supplied parameters list
 * `parameters`.
 *
 * @param [in] begin An iterator to the beginning of the sequence
 * @param [in] end An iterator to the end of the sequence
 * @param [out] parameters A reference to the parameters list to append
 * paramter values extracted to
 * @return An iterator to the character after the last parameter read
 */
template <typename InputIterator>
InputIterator extract_parameters(InputIterator begin, InputIterator end,
    parameter_list &parameters)
{
    InputIterator cursor;

    if (begin == end) {
        // error: expected non-zero length range
        return begin;
    }

    cursor = begin;
    std::pair<std::string,InputIterator> ret;

    /**
     * LWS
     * token
     * LWS
     * *(";" method-param)
     * LWS
     * ,=loop again
     */
    while (cursor != end) {
        std::string parameter_name;
        attribute_list attributes;

        // extract any stray whitespace
        cursor = http::parser::extract_all_lws(cursor,end);
        if (cursor == end) {break;}

        ret = http::parser::extract_token(cursor,end);

        if (ret.first.empty()) {
            // error: expected a token
            return begin;
        } else {
            parameter_name = ret.first;
            cursor = ret.second;
        }

        // Safe break point, insert parameter with blank attributes and exit
        cursor = http::parser::extract_all_lws(cursor,end);
        if (cursor == end) {
            //parameters[parameter_name] = attributes;
            parameters.push_back(std::make_pair(parameter_name,attributes));
            break;
        }

        // If there is an attribute list, read it in
        if (*cursor == ';') {
            InputIterator acursor;

            ++cursor;
            acursor = http::parser::extract_attributes(cursor,end,attributes);

            if (acursor == cursor) {
                // attribute extraction ended in syntax error
                return begin;
            }

            cursor = acursor;
        }

        // insert parameter into output list
        //parameters[parameter_name] = attributes;
        parameters.push_back(std::make_pair(parameter_name,attributes));

        cursor = http::parser::extract_all_lws(cursor,end);
        if (cursor == end) {break;}

        // if next char is ',' then read another parameter, else stop
        if (*cursor != ',') {
            break;
        }

        // advance past comma
        ++cursor;

        if (cursor == end) {
            // expected more bytes after a comma
            return begin;
        }
    }

    return cursor;
}

inline std::string strip_lws(std::string const & input) {
    std::string::const_iterator begin = extract_all_lws(input.begin(),input.end());
    if (begin == input.end()) {
        return std::string();
    }

    std::string::const_reverse_iterator rbegin = extract_all_lws(input.rbegin(),input.rend());
    if (rbegin == input.rend()) {
        return std::string();
    }

    return std::string(begin,rbegin.base());
}

/// Base HTTP parser
/**
 * Includes methods and data elements common to all types of HTTP messages such
 * as headers, versions, bodies, etc.
 */
class parser {
public:
    parser()
      : m_header_bytes(0)
      , m_body_bytes_needed(0)
      , m_body_bytes_max(max_body_size)
      , m_body_encoding(body_encoding::unknown) {}
    
    /// Get the HTTP version string
    /**
     * @return The version string for this parser
     */
    std::string const & get_version() const {
        return m_version;
    }

    /// Set HTTP parser Version
    /**
     * Input should be in format: HTTP/x.y where x and y are positive integers.
     * @todo Does this method need any validation?
     *
     * @param [in] version The value to set the HTTP version to.
     */
    void set_version(std::string const & version);

    /// Get the value of an HTTP header
    /**
     * @todo Make this method case insensitive.
     *
     * @param [in] key The name/key of the header to get.
     * @return The value associated with the given HTTP header key.
     */
    std::string const & get_header(std::string const & key) const;

    /// Extract an HTTP parameter list from a parser header.
    /**
     * If the header requested doesn't exist or exists and is empty the
     * parameter list is valid (but empty).
     *
     * @param [in] key The name/key of the HTTP header to use as input.
     * @param [out] out The parameter list to store extracted parameters in.
     * @return Whether or not the input was a valid parameter list.
     */
    bool get_header_as_plist(std::string const & key, parameter_list & out)
        const;

    /// Append a value to an existing HTTP header
    /**
     * This method will set the value of the HTTP header `key` with the
     * indicated value. If a header with the name `key` already exists, `val`
     * will be appended to the existing value.
     *
     * @todo Make this method case insensitive.
     * @todo Should there be any restrictions on which keys are allowed?
     * @todo Exception free varient
     *
     * @see replace_header
     *
     * @param [in] key The name/key of the header to append to.
     * @param [in] val The value to append.
     */
    void append_header(std::string const & key, std::string const & val);

    /// Set a value for an HTTP header, replacing an existing value
    /**
     * This method will set the value of the HTTP header `key` with the
     * indicated value. If a header with the name `key` already exists, `val`
     * will replace the existing value.
     *
     * @todo Make this method case insensitive.
     * @todo Should there be any restrictions on which keys are allowed?
     * @todo Exception free varient
     *
     * @see append_header
     *
     * @param [in] key The name/key of the header to append to.
     * @param [in] val The value to append.
     */
    void replace_header(std::string const & key, std::string const & val);

    /// Remove a header from the parser
    /**
     * Removes the header entirely from the parser. This is different than
     * setting the value of the header to blank.
     *
     * @todo Make this method case insensitive.
     *
     * @param [in] key The name/key of the header to remove.
     */
    void remove_header(std::string const & key);

    /// Get HTTP body
    /**
     * Gets the body of the HTTP object
     *
     * @return The body of the HTTP message.
     */
    std::string const & get_body() const {
        return m_body;
    }

    /// Set body content
    /**
     * Set the body content of the HTTP response to the parameter string. Note
     * set_body will also set the Content-Length HTTP header to the appropriate
     * value. If you want the Content-Length header to be something else, do so
     * via replace_header("Content-Length") after calling set_body()
     *
     * @param value String data to include as the body content.
     */
    void set_body(std::string const & value);

    /// Get body size limit
    /**
     * Retrieves the maximum number of bytes to parse & buffer before canceling
     * a request.
     *
     * @since 0.5.0
     *
     * @return The maximum length of a message body.
     */
    size_t get_max_body_size() const {
        return m_body_bytes_max;
    }

    /// Set body size limit
    /**
     * Set the maximum number of bytes to parse and buffer before canceling a
     * request.
     *
     * @since 0.5.0
     *
     * @param value The size to set the max body length to.
     */
    void set_max_body_size(size_t value) {
        m_body_bytes_max = value;
    }

    /// Extract an HTTP parameter list from a string.
    /**
     * @param [in] in The input string.
     * @param [out] out The parameter list to store extracted parameters in.
     * @return Whether or not the input was a valid parameter list.
     */
    bool parse_parameter_list(std::string const & in, parameter_list & out)
        const;
protected:
    /// Process a header line
    /**
     * @todo Update this method to be exception free.
     *
     * @param [in] begin An iterator to the beginning of the sequence.
     * @param [in] end An iterator to the end of the sequence.
     */
    void process_header(std::string::iterator begin, std::string::iterator end);

    /// Prepare the parser to begin parsing body data
    /**
     * Inspects headers to determine if the message has a body that needs to be
     * read. If so, sets up the necessary state, otherwise returns false. If
     * this method returns true and loading the message body is desired call
     * `process_body` until it returns zero bytes or an error.
     *
     * Must not be called until after all headers have been processed.
     *
     * @since 0.5.0
     *
     * @return True if more bytes are needed to load the body, false otherwise.
     */
    bool prepare_body();

    /// Process body data
    /**
     * Parses body data.
     *
     * @since 0.5.0
     *
     * @param [in] begin An iterator to the beginning of the sequence.
     * @param [in] end An iterator to the end of the sequence.
     * @return The number of bytes processed
     */
    size_t process_body(char const * buf, size_t len);

    /// Check if the parser is done parsing the body
    /**
     * Behavior before a call to `prepare_body` is undefined.
     *
     * @since 0.5.0
     *
     * @return True if the message body has been completed loaded.
     */
    bool body_ready() const {
        return (m_body_bytes_needed == 0);
    }

    /// Generate and return the HTTP headers as a string
    /**
     * Each headers will be followed by the \r\n sequence including the last one.
     * A second \r\n sequence (blank header) is not appended by this method
     *
     * @return The HTTP headers as a string.
     */
    std::string raw_headers() const;

    std::string m_version;
    header_list m_headers;
    
    size_t                  m_header_bytes;
    
    std::string             m_body;
    size_t                  m_body_bytes_needed;
    size_t                  m_body_bytes_max;
    body_encoding::value    m_body_encoding;
};

} // namespace parser
} // namespace http
} // namespace websocketpp

#include <websocketpp/http/impl/parser.hpp>

#endif // HTTP_PARSER_HPP
